param([ValidateSet('gated_reheat_calibration','persistence_calibration')]
      [string]$Study='gated_reheat_calibration')
$ErrorActionPreference = 'Stop'
$repoRoot = [System.IO.Path]::GetFullPath((Join-Path $PSScriptRoot '..\..'))
$pythonPath = [System.IO.Path]::GetFullPath((Join-Path $repoRoot '..\.venv\Scripts\python.exe'))
if (-not (Test-Path -LiteralPath $pythonPath)) { throw 'Workspace Python missing' }
$runRoot = Join-Path $repoRoot ('experiments\runs\'+$Study)
New-Item -ItemType Directory -Path $runRoot -Force | Out-Null
$stopPath = Join-Path $runRoot 'supervisor.stop'
if (Test-Path -LiteralPath $stopPath) { Remove-Item -LiteralPath $stopPath }
$launchStamp = Get-Date -Format 'yyyyMMdd_HHmmss_fff'
$stdout = Join-Path $runRoot "supervisor_boot_$launchStamp.log"
$stderr = Join-Path $runRoot "supervisor_boot_${launchStamp}_error.log"
$proc = Start-Process -FilePath $pythonPath -ArgumentList @('-u','-m','hvaclab.supervisor','--study',$Study) `
    -WorkingDirectory $repoRoot -WindowStyle Hidden -RedirectStandardOutput $stdout `
    -RedirectStandardError $stderr -PassThru
Write-Output "Supervisor launcher PID: $($proc.Id)"
Write-Output "State: $runRoot\supervisor_status.json"
