"""
LEO5 cluster status — concise free-resource summary.

Usage:
    python Master/scripts/leo5_status.py          # human-readable (default)
    python Master/scripts/leo5_status.py --json    # JSON output
"""

import argparse
import json
import re
import subprocess
import sys
from datetime import datetime


def ssh_cmd(cmd):
    """Run a command on LEO5 via SSH, return stdout."""
    result = subprocess.run(
        ["ssh", "leo5", cmd],
        capture_output=True, text=True, timeout=30,
    )
    if result.returncode != 0:
        print(f"SSH error: {result.stderr.strip()}", file=sys.stderr)
        sys.exit(1)
    return result.stdout.strip()


def parse_sinfo(raw):
    """Parse sinfo output into node dicts."""
    lines = raw.splitlines()
    if not lines:
        return []
    # Header: NodeHost CPUsState Gres GresUsed FreeMem Memory StateLong Partition
    nodes = []
    for line in lines[1:]:
        parts = line.split()
        if len(parts) < 8:
            continue
        host = parts[0]
        # CPUs: A/I/O/T (allocated/idle/other/total)
        cpu_match = re.match(r'(\d+)/(\d+)/(\d+)/(\d+)', parts[1])
        if not cpu_match:
            continue
        cpus_alloc, cpus_idle, _, cpus_total = map(int, cpu_match.groups())

        # GRES: (null) or gpu:a30:2 etc
        gres = parts[2]
        gpu_type, gpu_total = None, 0
        gres_match = re.match(r'gpu:(\w+):(\d+)', gres)
        if gres_match:
            gpu_type = gres_match.group(1)
            gpu_total = int(gres_match.group(2))

        # GresUsed: gpu:0 or gpu:a30:1(IDX:0) etc
        gres_used = parts[3]
        gpu_used = 0
        used_match = re.search(r'gpu:(?:\w+:)?(\d+)', gres_used)
        if used_match:
            gpu_used = int(used_match.group(1))

        free_mem_mb = int(parts[4])
        total_mem_mb = int(parts[5])
        state = parts[6]
        partition = parts[7]

        # Skip duplicate entries (mem2000 partition shows same nodes)
        if partition == 'mem2000':
            continue

        nodes.append({
            'host': host,
            'cpus_idle': cpus_idle,
            'cpus_total': cpus_total,
            'gpu_type': gpu_type,
            'gpu_total': gpu_total,
            'gpu_free': gpu_total - gpu_used,
            'free_mem_gb': round(free_mem_mb / 1024),
            'total_mem_gb': round(total_mem_mb / 1024),
            'state': state,
        })
    return nodes


def parse_squeue(raw):
    """Parse squeue output into pending/running counts."""
    running = 0
    pending = 0
    pending_gpu = 0
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


def parse_my_jobs(raw):
    """Parse user's own jobs."""
    jobs = []
    for line in raw.splitlines():
        parts = line.split(None, 5)
        if len(parts) < 4:
            continue
        jobs.append({
            'id': parts[0],
            'name': parts[1],
            'state': parts[2],
            'time': parts[3],
            'gres': parts[4] if len(parts) > 4 else '',
            'node': parts[5] if len(parts) > 5 else '',
        })
    return jobs


def build_summary(nodes, queue, my_jobs):
    """Build concise free-resource summary."""
    # GPU summary
    gpu_types = {}
    gpu_free_nodes = []
    for n in nodes:
        if n['gpu_type']:
            t = n['gpu_type']
            if t not in gpu_types:
                gpu_types[t] = {'total': 0, 'free': 0}
            gpu_types[t]['total'] += n['gpu_total']
            gpu_types[t]['free'] += n['gpu_free']
            if n['gpu_free'] > 0:
                gpu_free_nodes.append(n)

    # CPU-only free
    cpu_free_cores = 0
    cpu_free_node_count = 0
    for n in nodes:
        if not n['gpu_type'] and n['cpus_idle'] > 0:
            cpu_free_cores += n['cpus_idle']
            cpu_free_node_count += 1

    # Load percentage
    total_cpus = sum(n['cpus_total'] for n in nodes)
    idle_cpus = sum(n['cpus_idle'] for n in nodes)
    load_pct = round((1 - idle_cpus / total_cpus) * 100) if total_cpus > 0 else 0

    return {
        'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M'),
        'load_pct': load_pct,
        'gpu': gpu_types,
        'gpu_free_nodes': gpu_free_nodes,
        'cpu_free_cores': cpu_free_cores,
        'cpu_free_node_count': cpu_free_node_count,
        'queue': queue,
        'my_jobs': my_jobs,
    }


def print_human(s):
    """Print concise human-readable summary."""
    print(f"LEO5 {s['timestamp']} | load {s['load_pct']}%")

    # GPU line
    gpu_parts = []
    for t in ['a100', 'a30', 'a40']:
        if t in s['gpu']:
            g = s['gpu'][t]
            gpu_parts.append(f"{t.upper()} {g['free']}/{g['total']} free")
    if gpu_parts:
        print(f"GPU free: {', '.join(gpu_parts)}")

    # Split GPU nodes: schedulable (have idle CPUs) vs GPU-free-but-CPU-full
    schedulable = [n for n in s['gpu_free_nodes'] if n['cpus_idle'] > 0]
    gpu_only = [n for n in s['gpu_free_nodes'] if n['cpus_idle'] == 0]

    if schedulable:
        print("  Ready (GPU+CPU free):")
        for n in schedulable:
            print(f"    {n['host']} {n['gpu_type'].upper()}x{n['gpu_free']} {n['cpus_idle']}cpu {n['free_mem_gb']}G")

    if gpu_only:
        # Summarize by GPU type
        by_type = {}
        for n in gpu_only:
            t = n['gpu_type']
            if t not in by_type:
                by_type[t] = {'count': 0, 'gpus': 0}
            by_type[t]['count'] += 1
            by_type[t]['gpus'] += n['gpu_free']
        parts = [f"{v['gpus']}x {t.upper()} on {v['count']} nodes" for t, v in by_type.items()]
        print(f"  GPU free but 0 idle CPU: {', '.join(parts)}")

    # CPU-only free
    if s['cpu_free_cores'] > 0:
        print(f"CPU-only free: {s['cpu_free_cores']} cores across {s['cpu_free_node_count']} nodes")

    # Queue
    q = s['queue']
    gpu_part = f" ({q['pending_gpu']} want GPU)" if q['pending_gpu'] > 0 else ""
    print(f"Queue: {q['pending']} pending{gpu_part}")

    # My jobs
    if s['my_jobs']:
        print("My jobs:")
        for j in s['my_jobs']:
            print(f"  {j['id']} {j['name']} {j['state']} {j['time']}")
    else:
        print("My jobs: none")


def main():
    parser = argparse.ArgumentParser(description='LEO5 cluster free-resource status')
    parser.add_argument('--json', action='store_true', help='Output JSON')
    args = parser.parse_args()

    sinfo_raw = ssh_cmd(
        "sinfo -O 'NodeHost,CPUsState,Gres,GresUsed,FreeMem,Memory,StateLong,Partition' --partition=std"
    )
    squeue_raw = ssh_cmd(
        "squeue --format='%u %T %b %D %C %m %l %r' --noheader"
    )
    my_jobs_raw = ssh_cmd(
        "squeue -u c7031336 --format='%i %j %T %M %b %R' --noheader"
    )

    nodes = parse_sinfo(sinfo_raw)
    queue = parse_squeue(squeue_raw)
    my_jobs = parse_my_jobs(my_jobs_raw)
    summary = build_summary(nodes, queue, my_jobs)

    if args.json:
        # Concise JSON: only schedulable nodes listed individually
        schedulable = [n for n in summary['gpu_free_nodes'] if n['cpus_idle'] > 0]
        gpu_only = [n for n in summary['gpu_free_nodes'] if n['cpus_idle'] == 0]
        # Summarize GPU-free-but-no-CPU by type
        blocked = {}
        for n in gpu_only:
            t = n['gpu_type']
            if t not in blocked:
                blocked[t] = 0
            blocked[t] += n['gpu_free']
        json_out = {
            'timestamp': summary['timestamp'],
            'load_pct': summary['load_pct'],
            'gpu': summary['gpu'],
            'ready': [
                {'host': n['host'], 'gpu': f"{n['gpu_type']}x{n['gpu_free']}", 'cpu_idle': n['cpus_idle'], 'mem_free_gb': n['free_mem_gb']}
                for n in schedulable
            ],
            'gpu_free_no_cpu': blocked,
            'cpu_free': {'cores': summary['cpu_free_cores'], 'nodes': summary['cpu_free_node_count']},
            'queue': summary['queue'],
            'my_jobs': summary['my_jobs'],
        }
        print(json.dumps(json_out, indent=2))
    else:
        print_human(summary)


if __name__ == '__main__':
    main()
