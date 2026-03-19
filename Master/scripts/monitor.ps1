# Live hardware monitor: GPU + CPU (per-core) + RAM, refreshes every 10 seconds
# Run with: pwsh Master\temp\monitor.ps1
# Stop with: Ctrl+C
while ($true) {
    Clear-Host
    Write-Host "=== $(Get-Date -Format 'HH:mm:ss') ===" -ForegroundColor Cyan
    Write-Host ""
    Write-Host "--- GPU ---" -ForegroundColor Yellow
    nvidia-smi --query-gpu=utilization.gpu,memory.used,memory.total,temperature.gpu --format=csv
    Write-Host ""
    Write-Host "--- RAM ---" -ForegroundColor Yellow
    $os = Get-CimInstance Win32_OperatingSystem
    $totalRAM = [math]::Round($os.TotalVisibleMemorySize / 1MB, 1)
    $usedRAM = [math]::Round(($os.TotalVisibleMemorySize - $os.FreePhysicalMemory) / 1MB, 1)
    Write-Host "RAM Used: $usedRAM / $totalRAM GB"
    Write-Host ""
    Write-Host "--- CPU per core ---" -ForegroundColor Yellow
    $cores = Get-CimInstance Win32_PerfFormattedData_PerfOS_Processor
    foreach ($core in $cores) {
        $name = $core.Name
        $pct = $core.PercentProcessorTime
        if ($name -eq '_Total') {
            Write-Host "  TOTAL : $pct %" -ForegroundColor Cyan
        } else {
            $bar = '#' * [math]::Floor($pct / 5)
            Write-Host ("  Core {0,2}: {1,3}%  {2}" -f $name, $pct, $bar)
        }
    }
    Start-Sleep 10
}
