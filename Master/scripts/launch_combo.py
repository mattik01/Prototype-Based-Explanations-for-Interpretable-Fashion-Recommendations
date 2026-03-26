"""
Launch run_combo.py as a detached Windows process that survives SSH disconnection.

Launches the training in the background, then tails the log file so you see live
output in your terminal. If SSH drops, the training keeps running — just reconnect
and tail the log file to pick up where you left off.

Usage (from repo root):
    python Master/scripts/launch_combo.py -m item_proto -d ml-1m
    python Master/scripts/launch_combo.py -m mf -d amazon2014 -s 9491758 --delay 2

    # Launch without tailing (fire-and-forget):
    python Master/scripts/launch_combo.py -m item_proto -d ml-1m --no-follow
"""

import argparse
import os
import subprocess
import sys
import time

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.abspath(os.path.join(SCRIPT_DIR, "..", ".."))
LOG_DIR = os.path.join(REPO_ROOT, "Master", "temp", "gpu_logs")

# Windows process creation flags for full detachment
DETACHED_PROCESS = 0x00000008
CREATE_NEW_PROCESS_GROUP = 0x00000200
CREATE_NO_WINDOW = 0x08000000


def _is_pid_alive(pid):
    """Check if a Windows process is still running."""
    try:
        result = subprocess.run(
            ["tasklist", "/FI", f"PID eq {pid}", "/NH"],
            capture_output=True, text=True, timeout=5,
        )
        return str(pid) in result.stdout
    except Exception:
        return True  # assume alive if check fails


def _tail_follow(log_file, pid, poll_interval=0.5):
    """Tail a log file until the process exits, like 'tail -f'."""
    # Wait briefly for the file to get initial content
    time.sleep(0.5)

    with open(log_file, "r") as f:
        check_counter = 0
        while True:
            line = f.readline()
            if line:
                print(line, end="", flush=True)
                check_counter = 0
            else:
                time.sleep(poll_interval)
                check_counter += 1
                # Check if process is still alive every ~5 seconds
                if check_counter >= int(5 / poll_interval):
                    check_counter = 0
                    if not _is_pid_alive(pid):
                        # Drain remaining lines
                        for remaining in f:
                            print(remaining, end="", flush=True)
                        print(f"\n{'=' * 60}")
                        print(f"Training process (PID {pid}) has finished.")
                        print(f"Full log: {log_file}")
                        return


def main():
    parser = argparse.ArgumentParser(
        description='Launch run_combo.py as a detached background process',
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument('--model', '-m', type=str, required=True)
    parser.add_argument('--dataset', '-d', type=str, required=True)
    parser.add_argument('--seed', '-s', type=int, default=None)
    parser.add_argument('--delay', type=float, default=0)
    parser.add_argument('--no-follow', action='store_true',
                        help='Launch and exit without tailing the log')

    args = parser.parse_args()

    os.makedirs(LOG_DIR, exist_ok=True)

    timestamp = time.strftime("%Y%m%d_%H%M%S")
    log_file = os.path.join(LOG_DIR, f"combo_{args.model}_{args.dataset}_{timestamp}.log")

    # Build the command
    combo_script = os.path.join(SCRIPT_DIR, "run_combo.py")
    cmd = [sys.executable, combo_script, "-m", args.model, "-d", args.dataset]
    if args.seed is not None:
        cmd += ["-s", str(args.seed)]
    if args.delay > 0:
        cmd += ["--delay", str(args.delay)]

    print(f"Launching detached process...")
    print(f"  Command:  {' '.join(cmd)}")
    print(f"  Log file: {log_file}")

    with open(log_file, "w") as log_fh:
        proc = subprocess.Popen(
            cmd,
            stdout=log_fh,
            stderr=subprocess.STDOUT,
            cwd=REPO_ROOT,
            creationflags=DETACHED_PROCESS | CREATE_NEW_PROCESS_GROUP | CREATE_NO_WINDOW,
        )

    print(f"  PID:      {proc.pid}")
    print(f"\nProcess launched. Training survives SSH disconnection.")

    if args.no_follow:
        print(f"\nReconnect and tail later:")
        print(f"  tail -f '{log_file}'")
        return

    # Follow the log in real-time — Ctrl+C stops tailing, not the training
    print(f"Tailing log (Ctrl+C to stop watching — training continues)...\n")
    print("=" * 60)

    try:
        _tail_follow(log_file, proc.pid)
    except KeyboardInterrupt:
        print(f"\n\nStopped tailing. Training still running (PID {proc.pid}).")
        print(f"Resume watching:  tail -f '{log_file}'")


if __name__ == "__main__":
    main()
