import csv, sys, os

def summarize(path, label, concurrency=16):
    with open(path) as f:
        rows = list(csv.DictReader(f))
    gpu_utils = [float(r['gpu_util_pct']) for r in rows]
    vram = [float(r['vram_used_mib']) for r in rows]
    cpu = [float(r['cpu_pct']) for r in rows]
    ram = [float(r['ram_used_mib']) for r in rows]
    gpu_temp = [float(r['gpu_temp_c']) for r in rows] if 'gpu_temp_c' in rows[0] else None
    from datetime import datetime
    dt0 = datetime.strptime(rows[0]['timestamp'], '%Y-%m-%d %H:%M:%S')
    dt1 = datetime.strptime(rows[-1]['timestamp'], '%Y-%m-%d %H:%M:%S')
    wall_sec = (dt1 - dt0).total_seconds()
    h = int(wall_sec // 3600)
    m = int((wall_sec % 3600) // 60)
    peak_vram = max(vram)
    safe_vram = (peak_vram - 512) / concurrency
    print(f'=== {label} ===')
    print(f'Wall time: {h}h {m:02d}m')
    print(f'Peak VRAM: {peak_vram:.0f} MiB')
    print(f'SAFE-ISH VRAM: {safe_vram:.0f} MiB')
    print(f'Avg GPU%: {sum(gpu_utils)/len(gpu_utils):.1f}')
    if gpu_temp:
        print(f'Peak GPU temp: {max(gpu_temp):.0f} °C')
        print(f'Avg GPU temp: {sum(gpu_temp)/len(gpu_temp):.0f} °C')
    print(f'Avg CPU%: {sum(cpu)/len(cpu):.1f}')
    print(f'Avg RAM: {sum(ram)/len(ram):.0f} MiB')
    print(f'Peak RAM: {max(ram):.0f} MiB')
    print()

log_dir = r'C:\Users\glaes\Desktop\github\Prototype-Based-Explanations-for-Interpretable-Fashion-Recommendation\Master\temp\gpu_logs'
summarize(os.path.join(log_dir, 'gpu_acf_ml-1m_s38210573.csv'), 'ACF ml-1m')
summarize(os.path.join(log_dir, 'gpu_20260325_074951.csv'), 'user_proto ml-1m (crashed)', concurrency=16)
