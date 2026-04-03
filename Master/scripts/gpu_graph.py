"""
Live GPU utilization graph in the terminal (no GUI needed — works over SSH).
Run:  python Master/scripts/gpu_graph.py
Stop: Ctrl+C

Samples every 10s, draws rolling ASCII charts (last hour) for GPU util, VRAM, and RAM.
Purely visual — no file I/O. For CSV logging, use gpu_sampler.py.
"""

import subprocess
import shutil
import time
import sys
from datetime import datetime

import psutil

INTERVAL_SEC = 10
WINDOW_SEC = 3600  # last hour
GRAPH_HEIGHT = 7   # rows for each chart

# ANSI helpers
CSI = "\033["
CLEAR = f"{CSI}2J{CSI}H"
GREEN = f"{CSI}32m"
BLUE = f"{CSI}34m"
CYAN = f"{CSI}36m"
MAGENTA = f"{CSI}35m"
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


def query_ram():
    """Return (ram_used_MiB, ram_total_MiB) or None on failure."""
    try:
        mem = psutil.virtual_memory()
        return mem.used / (1024 * 1024), mem.total / (1024 * 1024)
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

    gpu_pcts = []
    mem_pcts = []
    ram_pcts = []
    max_vram_pct = 0.0
    max_vram_mib = 0.0
    max_ram_pct = 0.0
    max_ram_mib = 0.0
    gpu_sum = 0.0
    ram_sum = 0.0
    sample_count = 0

    print(f"Starting GPU monitor (session {session_id})...")
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

            ram_result = query_ram()
            ram_used, ram_total = ram_result if ram_result else (0, 1)
            ram_pct = ram_used / ram_total * 100

            gpu_pcts.append(gpu_util)
            mem_pcts.append(mem_pct)
            ram_pcts.append(ram_pct)

            # track stats (across all time, not just window)
            sample_count += 1
            gpu_sum += gpu_util
            ram_sum += ram_pct
            if mem_pct > max_vram_pct:
                max_vram_pct = mem_pct
                max_vram_mib = mem_used
            if ram_pct > max_ram_pct:
                max_ram_pct = ram_pct
                max_ram_mib = ram_used

            # trim to window
            max_points = WINDOW_SEC // INTERVAL_SEC
            if len(gpu_pcts) > max_points:
                del gpu_pcts[:-max_points]
                del mem_pcts[:-max_points]
                del ram_pcts[:-max_points]

            # determine chart width from terminal
            term_width = shutil.get_terminal_size().columns
            chart_width = max(20, term_width - 12)

            now = datetime.now()
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

            ram_text = f"{BOLD}{ram_used:.0f}/{ram_total:.0f} MiB ({ram_pct:.1f}%){RESET}"
            output.extend(draw_chart(ram_pcts, chart_width, GRAPH_HEIGHT, MAGENTA, "RAM Usage", ram_text))
            output.append("")

            avg_gpu = gpu_sum / sample_count
            elapsed = sample_count * INTERVAL_SEC
            if elapsed >= 3600:
                elapsed_str = f"{elapsed // 3600}h{(elapsed % 3600) // 60:02d}m"
            elif elapsed >= 60:
                elapsed_str = f"{elapsed // 60}m{elapsed % 60:02d}s"
            else:
                elapsed_str = f"{elapsed}s"
            avg_ram = ram_sum / sample_count
            output.append(
                f"  {YELLOW}[{session_id}]{RESET}  "
                f"{YELLOW}Avg GPU: {avg_gpu:.1f}%{RESET}  │  "
                f"{YELLOW}Peak VRAM: {max_vram_mib:.0f} MiB ({max_vram_pct:.1f}%){RESET}  │  "
                f"{YELLOW}Peak RAM: {max_ram_mib:.0f} MiB ({max_ram_pct:.1f}%){RESET}  │  "
                f"{DIM}Uptime: {elapsed_str}{RESET}"
            )
            output.append(f"  {DIM}Sampling every {INTERVAL_SEC}s | Ctrl+C to stop{RESET}")

            sys.stdout.write("\n".join(output) + "\n")
            sys.stdout.flush()

            time.sleep(INTERVAL_SEC)

    except KeyboardInterrupt:
        print(f"\n{RESET}Stopped.")


if __name__ == "__main__":
    main()
