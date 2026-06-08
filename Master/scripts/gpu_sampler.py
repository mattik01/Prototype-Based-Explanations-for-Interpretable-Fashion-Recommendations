"""
Standalone GPU + system CSV logger — importable and runnable.

Usage as module:
    from Master.scripts.gpu_sampler import GpuSampler, query_gpu
    sampler = GpuSampler(log_dir="Master/temp/gpu_logs/")
    sampler.start()
    ...
    sampler.stop()

Standalone test:
    python Master/scripts/gpu_sampler.py
    (Ctrl+C to stop — prints summary)
"""

import csv
import os
import subprocess
import threading
import time
from datetime import datetime

import psutil


def query_gpu():
    """Return (gpu_util%, mem_used_MiB, mem_total_MiB, temp_C) or None on failure."""
    try:
        out = subprocess.check_output(
            ["nvidia-smi", "--query-gpu=utilization.gpu,memory.used,memory.total,temperature.gpu",
             "--format=csv,noheader,nounits"],
            text=True,
        ).strip()
        parts = [float(x) for x in out.split(",")]
        return parts[0], parts[1], parts[2], parts[3]
    except Exception:
        return None


_prev_cpu_times = None
_prev_wall_time = None


def query_system():
    """Return (cpu_percent, ram_used_MiB, ram_total_MiB).

    Measures only the current process tree (this process + all children),
    so results are isolated to our job even on shared nodes.
    CPU% is computed from cumulative CPU time delta between calls.
    """
    global _prev_cpu_times, _prev_wall_time

    try:
        proc = psutil.Process()
        children = proc.children(recursive=True)
        all_procs = [proc] + children

        ram_used = 0
        total_cpu_time = 0
        for p in all_procs:
            try:
                ram_used += p.memory_info().rss
                ct = p.cpu_times()
                total_cpu_time += ct.user + ct.system
            except (psutil.NoSuchProcess, psutil.AccessDenied):
                pass
        ram_used /= (1024 * 1024)

        now = time.monotonic()
        if _prev_cpu_times is not None and _prev_wall_time is not None:
            dt = now - _prev_wall_time
            cpu_pct = ((total_cpu_time - _prev_cpu_times) / dt * 100) if dt > 0 else 0
        else:
            cpu_pct = 0

        _prev_cpu_times = total_cpu_time
        _prev_wall_time = now

    except (psutil.NoSuchProcess, psutil.AccessDenied):
        ram_used = 0
        cpu_pct = 0

    ram_total = psutil.virtual_memory().total / (1024 * 1024)
    return cpu_pct, ram_used, ram_total


class GpuSampler:
    """Background CSV logger for GPU utilization, VRAM, RAM, CPU usage, and GPU temperature."""

    def __init__(self, log_dir, interval_sec=10, session_id=None):
        self.log_dir = log_dir
        self.interval_sec = interval_sec
        # Uniqueness must come from job identity, NOT wall-clock time: two jobs that
        # start in the same second on a shared log dir otherwise collide on the CSV
        # path (one renames it out from under the other). Key on SLURM_JOB_ID + PID.
        if session_id is None:
            ts = datetime.now().strftime("%Y%m%d_%H%M%S")
            job = os.environ.get("SLURM_JOB_ID", "local")
            session_id = f"{ts}_{job}_{os.getpid()}"
        self.session_id = session_id

        self._stop_event = threading.Event()
        self._thread = None
        self._csv_file = None
        self._csv_writer = None
        self._csv_path = None

        # stats
        self._start_time = None
        self._samples = 0
        self._gpu_sum = 0.0
        self._vram_sum = 0.0
        self._max_vram_mib = 0.0
        self._cpu_sum = 0.0
        self._ram_sum = 0.0
        self._max_ram_mib = 0.0
        self._gpu_temp_sum = 0.0
        self._max_gpu_temp_c = 0.0

    def start(self):
        """Open CSV and spawn daemon sampling thread."""
        os.makedirs(self.log_dir, exist_ok=True)
        self._csv_path = os.path.join(self.log_dir, f"gpu_{self.session_id}.csv")
        self._csv_file = open(self._csv_path, "w", newline="")
        self._csv_writer = csv.writer(self._csv_file)
        self._csv_writer.writerow([
            "timestamp", "gpu_util_pct", "vram_used_mib", "vram_total_mib", "vram_pct",
            "cpu_pct", "ram_used_mib", "ram_total_mib", "ram_pct", "gpu_temp_c",
        ])

        self._start_time = time.monotonic()
        self._stop_event.clear()
        self._thread = threading.Thread(target=self._sample_loop, daemon=True)
        self._thread.start()

    def stop(self):
        """Signal stop, join thread, close CSV. Returns csv_path."""
        self._stop_event.set()
        if self._thread is not None:
            self._thread.join(timeout=self.interval_sec + 5)
        if self._csv_file is not None:
            self._csv_file.close()
            self._csv_file = None
        return self._csv_path

    def rename_log(self, new_name):
        """Rename the CSV file (just the basename). Returns the (possibly unchanged) path.

        Defensive: never raise. The CSV is non-essential telemetry — a missing source
        file must not abort the caller (which would otherwise skip result-saving).
        """
        if self._csv_path is None or not os.path.exists(self._csv_path):
            return self._csv_path
        new_path = os.path.join(os.path.dirname(self._csv_path), new_name)
        try:
            os.rename(self._csv_path, new_path)
            self._csv_path = new_path
        except OSError as e:
            print(f"[gpu_sampler] rename_log failed ({type(e).__name__}: {e}); "
                  f"keeping original telemetry path.")
        return self._csv_path

    def get_summary(self):
        """Return dict with resource utilization stats."""
        duration = time.monotonic() - self._start_time if self._start_time else 0
        n = max(self._samples, 1)
        return {
            "max_vram_mib": self._max_vram_mib,
            "avg_gpu_util": self._gpu_sum / n,
            "avg_vram_mib": self._vram_sum / n,
            "avg_cpu_pct": self._cpu_sum / n,
            "avg_ram_mib": self._ram_sum / n,
            "max_ram_mib": self._max_ram_mib,
            "max_gpu_temp_c": self._max_gpu_temp_c,
            "avg_gpu_temp_c": self._gpu_temp_sum / n,
            "total_samples": self._samples,
            "duration_sec": duration,
        }

    def _sample_loop(self):
        """Sampling loop — uses Event.wait() for responsive shutdown."""
        while not self._stop_event.is_set():
            gpu_result = query_gpu()
            cpu_pct, ram_used, ram_total = query_system()
            ram_pct = ram_used / ram_total * 100 if ram_total > 0 else 0

            if gpu_result is not None:
                gpu_util, mem_used, mem_total, gpu_temp = gpu_result
                vram_pct = mem_used / mem_total * 100 if mem_total > 0 else 0
            else:
                gpu_util, mem_used, mem_total, vram_pct, gpu_temp = 0, 0, 0, 0, 0

            now = datetime.now()
            self._csv_writer.writerow([
                now.strftime("%Y-%m-%d %H:%M:%S"),
                f"{gpu_util:.1f}",
                f"{mem_used:.0f}",
                f"{mem_total:.0f}",
                f"{vram_pct:.1f}",
                f"{cpu_pct:.1f}",
                f"{ram_used:.0f}",
                f"{ram_total:.0f}",
                f"{ram_pct:.1f}",
                f"{gpu_temp:.0f}",
            ])
            self._csv_file.flush()

            self._samples += 1
            self._gpu_sum += gpu_util
            self._vram_sum += mem_used
            self._cpu_sum += cpu_pct
            self._ram_sum += ram_used
            self._gpu_temp_sum += gpu_temp
            if mem_used > self._max_vram_mib:
                self._max_vram_mib = mem_used
            if ram_used > self._max_ram_mib:
                self._max_ram_mib = ram_used
            if gpu_temp > self._max_gpu_temp_c:
                self._max_gpu_temp_c = gpu_temp

            self._stop_event.wait(self.interval_sec)


if __name__ == "__main__":
    script_dir = os.path.dirname(os.path.abspath(__file__))
    log_dir = os.path.join(script_dir, "..", "temp", "gpu_logs")

    sampler = GpuSampler(log_dir=log_dir)
    print(f"GPU sampler started (session {sampler.session_id})")
    print(f"Logging to: {os.path.join(log_dir, f'gpu_{sampler.session_id}.csv')}")
    print("Press Ctrl+C to stop.\n")
    sampler.start()

    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        csv_path = sampler.stop()
        summary = sampler.get_summary()
        dur = summary["duration_sec"]
        if dur >= 3600:
            dur_str = f"{int(dur)//3600}h{(int(dur)%3600)//60:02d}m"
        elif dur >= 60:
            dur_str = f"{int(dur)//60}m{int(dur)%60:02d}s"
        else:
            dur_str = f"{dur:.0f}s"

        print(f"\n{'='*50}")
        print(f"  Resource Sampler Summary")
        print(f"{'='*50}")
        print(f"  Peak VRAM:       {summary['max_vram_mib']:.0f} MiB")
        print(f"  Avg VRAM:        {summary['avg_vram_mib']:.0f} MiB")
        print(f"  Avg GPU util:    {summary['avg_gpu_util']:.1f}%")
        print(f"  Peak GPU temp:   {summary['max_gpu_temp_c']:.0f} °C")
        print(f"  Avg GPU temp:    {summary['avg_gpu_temp_c']:.0f} °C")
        print(f"  Peak RAM:        {summary['max_ram_mib']:.0f} MiB")
        print(f"  Avg RAM:         {summary['avg_ram_mib']:.0f} MiB")
        print(f"  Avg CPU:         {summary['avg_cpu_pct']:.1f}%")
        print(f"  Samples:         {summary['total_samples']}")
        print(f"  Duration:        {dur_str}")
        print(f"  CSV:             {csv_path}")
        print(f"{'='*50}")
