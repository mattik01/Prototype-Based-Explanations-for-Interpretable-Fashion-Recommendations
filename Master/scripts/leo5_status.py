"""
LEO5 cluster status — what can actually be placed right now, and when held GPUs free up.

Usage:
    python Master/scripts/leo5_status.py                      # human-readable (default)
    python Master/scripts/leo5_status.py --json               # JSON output
    python Master/scripts/leo5_status.py --need-mem-gb 44 --need-cpus 18   # placeability thresholds

Rewritten 2026-10-26 (wave-1 submission sitting). Two things the old version got wrong:

1. Memory. It showed sinfo's FreeMem (OS-level free RAM), which is not what SLURM
   schedules on. The `std` partition is shared with CPU-only jobs that reserve RAM and
   cores on GPU nodes, so a GPU can sit idle on a node whose SLURM-allocatable memory is
   gone (2026-10-26: six A100s idle, 1–13 GB unallocated per node, FreeMem ~200 GB).
   This version reports Memory - AllocMem ("unalloc"), the number a --mem request is
   checked against.
2. "Free GPU" is not "placeable". A node is placeable only if a free GPU, enough idle
   CPUs (2 per requested --cpus, see run_combo.slurm F-S0-14 note) and enough
   unallocated RAM coincide. The script applies thresholds (defaults = the smallest
   production job shape: --cpus 6 → 12 CPUs, --mem 36G) and says which GPU-free nodes
   are blocked and why. It also prints a release horizon: how many held GPUs of each
   type come free within which time, from the holders' TimeLeft.
"""

import argparse
import json
import math
import re
import subprocess
import sys
from collections import Counter, defaultdict
from datetime import datetime

USER = 'c7031336'
GPU_MEM_GB = {'a100': 40, 'a100_80': 80, 'a30': 24, 'a40': 48}   # verified from job logs 2026-10-26


def ssh_cmd(cmd):
    """Run a command on LEO5 via SSH, return stdout."""
    result = subprocess.run(
        ["ssh", "-o", "ConnectTimeout=15", "-o", "BatchMode=yes", "leo5", cmd],
        capture_output=True, text=True, timeout=45,
    )
    if result.returncode != 0:
        print(f"SSH error: {result.stderr.strip()}", file=sys.stderr)
        print("(VPN down? ssh leo5 must work non-interactively.)", file=sys.stderr)
        sys.exit(1)
    return result.stdout.strip()


def parse_timeleft(s):
    """SLURM TimeLeft ('D-HH:MM:SS', 'HH:MM:SS', 'MM:SS') → seconds, or None."""
    if not s or not s[0].isdigit():
        return None
    days = 0
    if '-' in s:
        d, s = s.split('-', 1)
        days = int(d)
    parts = [int(p) for p in s.split(':')]
    while len(parts) < 3:
        parts.insert(0, 0)
    h, m, sec = parts
    return days * 86400 + h * 3600 + m * 60 + sec


def fmt_hours(h):
    return f"{h // 24}d{h % 24:02d}h" if h >= 24 else f"{h}h"


def parse_sinfo(raw):
    """sinfo -O NodeHost,CPUsState,Gres,GresUsed,AllocMem,Memory,StateLong,Partition → node dicts."""
    nodes = []
    for line in raw.splitlines()[1:]:
        parts = line.split()
        if len(parts) < 8:
            continue
        host = parts[0]
        cpu_match = re.match(r'(\d+)/(\d+)/(\d+)/(\d+)', parts[1])
        if not cpu_match:
            continue
        cpus_alloc, cpus_idle, _, cpus_total = map(int, cpu_match.groups())

        gres = parts[2]
        gpu_type, gpu_total = None, 0
        gres_match = re.match(r'gpu:(\w+):(\d+)', gres)
        if gres_match:
            gpu_type = gres_match.group(1)
            gpu_total = int(gres_match.group(2))

        used_match = re.search(r'gpu:(?:\w+:)?(\d+)', parts[3])
        gpu_used = int(used_match.group(1)) if used_match else 0

        alloc_mem_mb = int(parts[4]) if parts[4].isdigit() else 0
        total_mem_mb = int(parts[5]) if parts[5].isdigit() else 0
        state = parts[6]
        partition = parts[7]
        if partition == 'mem2000':
            continue

        nodes.append({
            'host': host,
            'cpus_idle': cpus_idle,
            'cpus_total': cpus_total,
            'gpu_type': gpu_type,
            'gpu_total': gpu_total,
            'gpu_free': max(0, gpu_total - gpu_used),
            'unalloc_mem_gb': max(0, (total_mem_mb - alloc_mem_mb) // 1024),
            'total_mem_gb': round(total_mem_mb / 1024),
            'state': state,
        })
    return nodes


def parse_queue(raw):
    """squeue (all users) '%u %T %b' lines → counts."""
    running = pending = pending_gpu = 0
    for line in raw.splitlines():
        parts = line.split()
        if len(parts) < 2:
            continue
        state = parts[1]
        tres = parts[2] if len(parts) > 2 else ''
        if state == 'RUNNING':
            running += 1
        elif state == 'PENDING':
            pending += 1
            if 'gpu' in tres.lower():
                pending_gpu += 1
    return {'running': running, 'pending': pending, 'pending_gpu': pending_gpu}


def parse_gpu_holders(raw, node_type):
    """squeue -t R -O TimeLeft,NodeList,tres-alloc lines → {gpu_type: [seconds_left, ...]} (one entry per held GPU)."""
    held = defaultdict(list)
    for line in raw.splitlines():
        parts = line.split(None, 2)
        if len(parts) < 3:
            continue
        timeleft, nodelist, tres = parts
        m = re.search(r'gres/gpu=(\d+)', tres)
        if not m:
            continue
        n_gpus = int(m.group(1))
        gtype = node_type.get(nodelist)
        if gtype is None:
            tm = re.search(r'gres/gpu:(\w+)=', tres)
            gtype = tm.group(1) if tm else 'unknown'
        held[gtype].extend([parse_timeleft(timeleft)] * n_gpus)
    return held


def release_horizon(held):
    """Per GPU type: how many GPUs are held and in which 6-hour bins they come free."""
    out = {}
    for gtype, secs in held.items():
        known = sorted(s for s in secs if s is not None)
        bins = Counter(math.ceil(s / 21600) * 6 for s in known)
        out[gtype] = {
            'held': len(secs),
            'releases': [{'within': fmt_hours(h), 'gpus': c} for h, c in sorted(bins.items())],
        }
    return out


def parse_my_jobs(raw):
    jobs = []
    for line in raw.splitlines():
        parts = line.split(None, 5)
        if len(parts) < 4:
            continue
        jobs.append({
            'id': parts[0], 'name': parts[1], 'state': parts[2], 'time': parts[3],
            'gres': parts[4] if len(parts) > 4 else '',
            'where': parts[5] if len(parts) > 5 else '',
        })
    return jobs


def build_summary(nodes, queue, held, my_jobs, need_cpus, need_mem_gb):
    gpu_types = {}
    gpu_free_nodes = []
    for n in nodes:
        if n['gpu_type']:
            t = gpu_types.setdefault(n['gpu_type'], {'total': 0, 'free': 0})
            t['total'] += n['gpu_total']
            t['free'] += n['gpu_free']
            if n['gpu_free'] > 0:
                gpu_free_nodes.append(n)

    placeable, blocked = [], []
    for n in gpu_free_nodes:
        reasons = []
        if n['cpus_idle'] < need_cpus:
            reasons.append(f"cpu {n['cpus_idle']}<{need_cpus}")
        if n['unalloc_mem_gb'] < need_mem_gb:
            reasons.append(f"mem {n['unalloc_mem_gb']}G<{need_mem_gb}G")
        (blocked if reasons else placeable).append({**n, 'why': ', '.join(reasons)})

    total_cpus = sum(n['cpus_total'] for n in nodes)
    idle_cpus = sum(n['cpus_idle'] for n in nodes)
    load_pct = round((1 - idle_cpus / total_cpus) * 100) if total_cpus else 0

    return {
        'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M'),
        'load_pct': load_pct,
        'thresholds': {'need_cpus': need_cpus, 'need_mem_gb': need_mem_gb},
        'gpu': gpu_types,
        'placeable': placeable,
        'blocked': blocked,
        'horizon': release_horizon(held),
        'queue': queue,
        'my_jobs': my_jobs,
    }


def print_human(s):
    print(f"LEO5 {s['timestamp']} | load {s['load_pct']}%")
    parts = []
    for t in ['a100', 'a100_80', 'a30', 'a40']:
        if t in s['gpu']:
            g = s['gpu'][t]
            parts.append(f"{t.upper()} {g['free']}/{g['total']} idle ({GPU_MEM_GB.get(t, '?')} GB)")
    print(f"GPUs: {', '.join(parts)}   (idle != placeable)")

    th = s['thresholds']
    print(f"Placeable now (free GPU + >= {th['need_cpus']} idle CPU + >= {th['need_mem_gb']}G unallocated RAM):")
    if s['placeable']:
        for n in s['placeable']:
            print(f"  {n['host']} {n['gpu_type'].upper()}x{n['gpu_free']}  {n['cpus_idle']} cpu idle  {n['unalloc_mem_gb']}G unalloc")
    else:
        print("  none")
    if s['blocked']:
        print("GPU idle but node blocked:")
        for n in s['blocked']:
            print(f"  {n['host']} {n['gpu_type'].upper()}x{n['gpu_free']}  {n['cpus_idle']} cpu idle  {n['unalloc_mem_gb']}G unalloc  ({n['why']})")

    if s['horizon']:
        print("Release horizon (held GPUs, by holders' time left):")
        for t in ['a100', 'a100_80', 'a30', 'a40', 'unknown']:
            if t in s['horizon']:
                h = s['horizon'][t]
                rel = ', '.join(f"{r['gpus']} within {r['within']}" for r in h['releases'][:5])
                print(f"  {t.upper()}: {h['held']} held; {rel}")

    q = s['queue']
    gpu_part = f" ({q['pending_gpu']} want a GPU)" if q['pending_gpu'] else ""
    print(f"Queue: {q['running']} running, {q['pending']} pending{gpu_part}")

    if s['my_jobs']:
        print(f"My jobs ({len(s['my_jobs'])}):")
        for j in s['my_jobs']:
            print(f"  {j['id']} {j['name']} {j['state']} {j['time']} {j['gres']} {j['where']}")
    else:
        print("My jobs: none")


def main():
    parser = argparse.ArgumentParser(description='LEO5 placeability + GPU release horizon')
    parser.add_argument('--json', action='store_true', help='Output JSON')
    parser.add_argument('--need-cpus', type=int, default=12,
                        help='idle CPUs a node must have to count as placeable (2 x --cpus; default 12)')
    parser.add_argument('--need-mem-gb', type=int, default=36,
                        help='SLURM-unallocated GB a node must have to count as placeable (default 36)')
    args = parser.parse_args()

    sinfo_raw = ssh_cmd(
        "sinfo -O 'NodeHost,CPUsState,Gres,GresUsed,AllocMem,Memory,StateLong,Partition' --partition=std"
    )
    queue_raw = ssh_cmd("squeue --format='%u %T %b' --noheader")
    holders_raw = ssh_cmd("squeue -t R -h -O 'TimeLeft:14,NodeList:18,tres-alloc:120'")
    my_jobs_raw = ssh_cmd(f"squeue -u {USER} --format='%i %j %T %M %b %R' --noheader")

    nodes = parse_sinfo(sinfo_raw)
    node_type = {n['host']: n['gpu_type'] for n in nodes if n['gpu_type']}
    summary = build_summary(nodes, parse_queue(queue_raw), parse_gpu_holders(holders_raw, node_type),
                            parse_my_jobs(my_jobs_raw), args.need_cpus, args.need_mem_gb)

    if args.json:
        slim = {k: v for k, v in summary.items()}
        slim['placeable'] = [{'host': n['host'], 'gpu': f"{n['gpu_type']}x{n['gpu_free']}",
                              'cpu_idle': n['cpus_idle'], 'unalloc_mem_gb': n['unalloc_mem_gb']} for n in summary['placeable']]
        slim['blocked'] = [{'host': n['host'], 'gpu': f"{n['gpu_type']}x{n['gpu_free']}",
                            'cpu_idle': n['cpus_idle'], 'unalloc_mem_gb': n['unalloc_mem_gb'], 'why': n['why']} for n in summary['blocked']]
        print(json.dumps(slim, indent=2))
    else:
        print_human(summary)


if __name__ == '__main__':
    main()
