
#### GPU monitor
nvidia-smi --query-gpu=utilization.gpu,memory.used,memory.total --format=csv -l 1    

#### pwsh hardware monitor
pwsh Master\scripts\monitor.ps1