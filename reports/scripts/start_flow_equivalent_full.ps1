$ErrorActionPreference='Stop'
$repoRoot=[System.IO.Path]::GetFullPath((Join-Path $PSScriptRoot '..\..'))
$pythonPath=[System.IO.Path]::GetFullPath((Join-Path $repoRoot '..\.venv\Scripts\python.exe'))
$runRoot=Join-Path $repoRoot 'experiments\runs\flow_equivalent_full'
New-Item -ItemType Directory -Path $runRoot -Force | Out-Null
$stamp=Get-Date -Format 'yyyyMMdd_HHmmss_fff'
$proc=Start-Process -FilePath $pythonPath -ArgumentList @('-u','-m','hvaclab.flow_equivalent_full_queue') `
    -WorkingDirectory $repoRoot -WindowStyle Hidden -PassThru `
    -RedirectStandardOutput (Join-Path $runRoot "queue_boot_$stamp.log") `
    -RedirectStandardError (Join-Path $runRoot "queue_boot_${stamp}_error.log")
Write-Output "Flow equivalent full launcher PID: $($proc.Id)"
