# Phase 2.3 — Full replication: 5 models × 2 datasets, single seed
# Run with: pwsh Master\scripts\replication.ps1
# Runs sequentially (Ray Tune needs exclusive GPU access)
# Ctrl+C to abort — safe to resume later from where you left off

param(
    [string]$StartFrom = ""  # Optional: skip to a specific model, e.g. "user_proto"
)

$models = @("mf", "user_item_proto", "user_proto", "item_proto", "acf")
$datasets = @("ml-1m", "amazon2014")

$skip = $StartFrom -ne ""
$total = $models.Count * $datasets.Count
$current = 0

foreach ($dataset in $datasets) {
    foreach ($model in $models) {
        if ($skip -and $model -ne $StartFrom) { continue }
        $skip = $false
        $current++
        Write-Host ""
        Write-Host "========================================" -ForegroundColor Cyan
        Write-Host "  [$current/$total] $model on $dataset" -ForegroundColor Cyan
        Write-Host "  Started: $(Get-Date -Format 'yyyy-MM-dd HH:mm:ss')" -ForegroundColor Cyan
        Write-Host "========================================" -ForegroundColor Cyan
        Write-Host ""

        python start.py -m $model -d $dataset

        if ($LASTEXITCODE -ne 0) {
            Write-Host "FAILED: $model on $dataset (exit code $LASTEXITCODE)" -ForegroundColor Red
            Write-Host "Fix the issue and re-run with: pwsh Master\scripts\replication.ps1 -StartFrom $model" -ForegroundColor Yellow
            exit 1
        }

        Write-Host ""
        Write-Host "  Completed: $(Get-Date -Format 'yyyy-MM-dd HH:mm:ss')" -ForegroundColor Green
    }
}

Write-Host ""
Write-Host "========================================" -ForegroundColor Green
Write-Host "  All replication runs complete!" -ForegroundColor Green
Write-Host "  $(Get-Date -Format 'yyyy-MM-dd HH:mm:ss')" -ForegroundColor Green
Write-Host "========================================" -ForegroundColor Green
