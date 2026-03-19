"""
Live GPU utilization graph in the terminal (no GUI needed — works over SSH).
Run:  python Master/scripts/gpu_graph.py
Stop: Ctrl+C

Samples every 60s, draws a rolling ASCII chart (last hour), logs to CSV.
"""

import subprocess
import shutil
import time
import sys
import os
import csv
from datetime import datetime

INTERVAL_SEC = 60
WINDOW_SEC = 3600  # last hour
GRAPH_HEIGHT = 7   # rows for each chart

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
LOG_DIR = os.path.join(SCRIPT_DIR, "..", "temp", "gpu_logs")

# ANSI helpers
CSI = "\033["
CLEAR = f"{CSI}2J{CSI}H"
GREEN = f"{CSI}32m"
BLUE = f"{CSI}34m"
CYAN = f"{CSI}36m"
YELLOW = f"{CSI}33m"
DIM = f"{CSI}2m"
RESET = f"{CSI}0m"
BOLD = f"{CSI}1m"

BLOCK_CHARS = " ▁▂▃▄▅▆▇█"


def query_gpu():
    """Return (gpu_util%, mem_used_MiB, mem_total_MiB) or None on failure."""
    try:
        out = subprocess.check_output(
            ["nvidia-smi", "--query-gpu=utilization.gpu,memory.used,memory.total",
             "--format=csv,noheader,nounits"],
            text=True,
        ).strip()
        parts = [float(x) for x in out.split(",")]
        return parts[0], parts[1], parts[2]
    except Exception:
        return None


def draw_chart(values, width, height, color, label, current_text):
    """Render an ASCII area chart, returns list of strings."""
    lines = []

    # downsample if more data points than chart width
    if len(values) > width:
        data = []
        chunk = len(values) / width
        for i in range(width):
            start = int(i * chunk)
            end = int((i + 1) * chunk)
            data.append(max(values[start:end]))
    else:
        data = list(values)

    lines.append(f"  {BOLD}{color}{label}{RESET}  {current_text}")

    for row in range(height, 0, -1):
        threshold = (row / height) * 100
        if row == height:
            lbl = "100%"
        elif row == 1:
            lbl = "  0%"
        else:
            lbl = "    "

        line = f"  {DIM}{lbl}{RESET} {DIM}│{RESET}"

        pad = width - len(data)
        line += " " * pad

        for val in data:
            if val >= threshold:
                line += f"{color}█{RESET}"
            elif val >= threshold - (100 / height) * 0.5:
                frac = (val - (threshold - 100 / height)) / (100 / height)
                idx = int(frac * (len(BLOCK_CHARS) - 1))
                idx = max(0, min(len(BLOCK_CHARS) - 1, idx))
                line += f"{color}{BLOCK_CHARS[idx]}{RESET}"
            else:
                line += " "

        lines.append(line)

    # x-axis
    axis = f"  {DIM}     └{'─' * width}{RESET}"
    lines.append(axis)

    # time labels
    total_sec = len(values) * INTERVAL_SEC
    if total_sec >= 3600:
        t_label = f"{total_sec // 3600}h{(total_sec % 3600) // 60:02d}m"
    elif total_sec >= 60:
        t_label = f"{total_sec // 60}m"
    else:
        t_label = f"{total_sec}s"
    time_label = f"  {DIM}      -{t_label}{' ' * max(0, width - 4 - len(t_label))}now{RESET}"
    lines.append(time_label)

    return lines


def main():
    session_id = datetime.now().strftime("%Y%m%d_%H%M%S")

    # setup CSV logging
    os.makedirs(LOG_DIR, exist_ok=True)
    csv_path = os.path.join(LOG_DIR, f"gpu_{session_id}.csv")
    csv_file = open(csv_path, "w", newline="")
    csv_writer = csv.writer(csv_file)
    csv_writer.writerow(["timestamp", "gpu_util_pct", "vram_used_mib", "vram_total_mib", "vram_pct"])

    gpu_pcts = []
    mem_pcts = []
    max_vram_pct = 0.0
    max_vram_mib = 0.0
    gpu_sum = 0.0
    sample_count = 0

    print(f"Starting GPU monitor (session {session_id})...")
    print(f"Logging to: {csv_path}")
    time.sleep(0.5)

    try:
        while True:
            result = query_gpu()
            if result is None:
                print("nvidia-smi failed, retrying...")
                time.sleep(INTERVAL_SEC)
                continue

            gpu_util, mem_used, mem_total = result
            mem_pct = mem_used / mem_total * 100
            now = datetime.now()

            gpu_pcts.append(gpu_util)
            mem_pcts.append(mem_pct)

            # log to CSV
            csv_writer.writerow([
                now.strftime("%Y-%m-%d %H:%M:%S"),
                f"{gpu_util:.1f}",
                f"{mem_used:.0f}",
                f"{mem_total:.0f}",
                f"{mem_pct:.1f}",
            ])
            csv_file.flush()

            # track stats (across all time, not just window)
            sample_count += 1
            gpu_sum += gpu_util
            if mem_pct > max_vram_pct:
                max_vram_pct = mem_pct
                max_vram_mib = mem_used

            # trim to window
            max_points = WINDOW_SEC // INTERVAL_SEC
            if len(gpu_pcts) > max_points:
                del gpu_pcts[:-max_points]
                del mem_pcts[:-max_points]

            # determine chart width from terminal
            term_width = shutil.get_terminal_size().columns
            chart_width = max(20, term_width - 12)

            now_str = now.strftime("%H:%M:%S")
            output = [CLEAR]
            output.append(f"  {BOLD}{CYAN}═══ GPU Monitor  {now_str}  [{session_id}] ═══{RESET}")
            output.append("")

            gpu_text = f"{BOLD}{gpu_util:5.1f}%{RESET}"
            output.extend(draw_chart(gpu_pcts, chart_width, GRAPH_HEIGHT, GREEN, "GPU Utilization", gpu_text))
            output.append("")

            mem_text = f"{BOLD}{mem_used:.0f}/{mem_total:.0f} MiB ({mem_pct:.1f}%){RESET}"
            output.extend(draw_chart(mem_pcts, chart_width, GRAPH_HEIGHT, BLUE, "VRAM Usage", mem_text))
            output.append("")

            avg_gpu = gpu_sum / sample_count
            elapsed = sample_count * INTERVAL_SEC
            if elapsed >= 3600:
                elapsed_str = f"{elapsed // 3600}h{(elapsed % 3600) // 60:02d}m"
            elif elapsed >= 60:
                elapsed_str = f"{elapsed // 60}m{elapsed % 60:02d}s"
            else:
                elapsed_str = f"{elapsed}s"
            output.append(
                f"  {YELLOW}[{session_id}]{RESET}  "
                f"{YELLOW}Avg GPU: {avg_gpu:.1f}%{RESET}  │  "
                f"{YELLOW}Peak VRAM: {max_vram_mib:.0f} MiB ({max_vram_pct:.1f}%){RESET}  │  "
                f"{DIM}Uptime: {elapsed_str}{RESET}"
            )
            output.append(f"  {DIM}Sampling every {INTERVAL_SEC}s | CSV: {csv_path} | Ctrl+C to stop{RESET}")

            sys.stdout.write("\n".join(output) + "\n")
            sys.stdout.flush()

            time.sleep(INTERVAL_SEC)

    except KeyboardInterrupt:
        csv_file.close()
        print(f"\n{RESET}Stopped. CSV saved: {csv_path}")


if __name__ == "__main__":
    main()
