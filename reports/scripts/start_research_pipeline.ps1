$ErrorActionPreference='Stop'
$repoRoot=[System.IO.Path]::GetFullPath((Join-Path $PSScriptRoot '..\..'))
$pythonPath=[System.IO.Path]::GetFullPath((Join-Path $repoRoot '..\.venv\Scripts\python.exe'))
foreach ($study in @('curve_refinement','heating_damper')) {
    $runRoot=Join-Path $repoRoot ('experiments\runs\'+$study)
    New-Item -ItemType Directory -Path $runRoot -Force | Out-Null
    $stamp=Get-Date -Format 'yyyyMMdd_HHmmss_fff'
    $proc=Start-Process -FilePath $pythonPath -ArgumentList @('-u','-m','hvaclab.pipeline_queue','--study',$study) `
        -WorkingDirectory $repoRoot -WindowStyle Hidden -PassThru `
        -RedirectStandardOutput (Join-Path $runRoot "queue_boot_$stamp.log") `
        -RedirectStandardError (Join-Path $runRoot "queue_boot_${stamp}_error.log")
    Write-Output "$study queue launcher PID: $($proc.Id)"
}
