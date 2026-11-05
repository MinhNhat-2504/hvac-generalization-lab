param([string]$RunRelativePath = 'experiments\runs\delay_compensation',
      [string]$StatusFileName = 'status.json')
$ErrorActionPreference = 'Stop'
$repoRoot = [System.IO.Path]::GetFullPath((Join-Path $PSScriptRoot '..\..'))
$runRoot = [System.IO.Path]::GetFullPath((Join-Path $repoRoot $RunRelativePath))
if (-not $runRoot.StartsWith($repoRoot + [System.IO.Path]::DirectorySeparatorChar, [System.StringComparison]::OrdinalIgnoreCase)) {
    throw 'Run directory must be inside the repository'
}
if ([System.IO.Path]::GetFileName($StatusFileName) -ne $StatusFileName) {
    throw 'StatusFileName must be a file name inside the run directory'
}
$statusPath = Join-Path $runRoot $StatusFileName
$noticePath = Join-Path $runRoot 'completion_notice.json'
$statePath = Join-Path $runRoot 'notification_watcher.json'

# Prevent duplicate watchers for this run. The OS releases the mutex if this
# process exits; it does not alter any experiment sources or results.
$mutexName = 'Local\HVACNotice_' + ($RunRelativePath -replace '[^a-zA-Z0-9]', '_')
$mutex = New-Object System.Threading.Mutex($false, $mutexName)
if (-not $mutex.WaitOne(0)) { exit 0 }
try {
    @{status='watching'; pid=$PID; started_at=(Get-Date).ToString('o')} |
        ConvertTo-Json | Set-Content -LiteralPath $statePath -Encoding UTF8
    while ($true) {
        try {
            $state = Get-Content -LiteralPath $statusPath -Raw | ConvertFrom-Json
        } catch {
            Start-Sleep -Seconds 20
            continue
        }
        if ($state.status -in @('complete', 'failed', 'interrupted')) { break }
        Start-Sleep -Seconds 20
    }
    if (Test-Path -LiteralPath $noticePath) {
        try {
            $previous = Get-Content -LiteralPath $noticePath -Raw | ConvertFrom-Json
            if ($previous.run_updated_at -eq $state.updated_at) { exit 0 }
        } catch {}
    }
    $message = "HVAC experiment ($RunRelativePath): $($state.status).`r`n"
    $message += "Completed rollouts: $($state.completed).`r`n"
    if ($state.decision.stop_reason) { $message += "$($state.decision.stop_reason)`r`n" }
    if ($state.decision.full_training) {
        $message += "Full training change: $([math]::Round($state.decision.full_training.change_pct, 2))%.`r`n"
        $message += "Full dev change: $([math]::Round($state.decision.full_dev.change_pct, 2))%.`r`n"
    }
    if ($state.decision.submission_notebook) {
        $message += "`r`nVerified submission notebook: $($state.decision.submission_notebook)`r`n"
    }
    $message += "`r`nResults: $RunRelativePath`r`nPaper: reports\paper\main.pdf"
    @{status=$state.status; run_updated_at=$state.updated_at; message=$message;
      recorded_at=(Get-Date).ToString('o')} |
        ConvertTo-Json | Set-Content -LiteralPath $noticePath -Encoding UTF8
    # Allow the final automatic PDF build to finish before displaying paths.
    Start-Sleep -Seconds 10
    Add-Type -AssemblyName System.Windows.Forms
    $notification = New-Object System.Windows.Forms.NotifyIcon
    try {
        $notification.Icon = [System.Drawing.SystemIcons]::Information
        $notification.Text = 'HVAC experiment'
        $notification.Visible = $true
        $balloonMessage = $message
        if ($balloonMessage.Length -gt 1000) { $balloonMessage = $balloonMessage.Substring(0, 1000) }
        $notification.ShowBalloonTip(15000, 'HVAC experiment finished', $balloonMessage,
            [System.Windows.Forms.ToolTipIcon]::Info)
        Start-Sleep -Seconds 15
    } finally {
        $notification.Visible = $false
        $notification.Dispose()
    }
    @{status='notification_display_attempted'; pid=$PID; updated_at=(Get-Date).ToString('o')} |
        ConvertTo-Json | Set-Content -LiteralPath $statePath -Encoding UTF8
} catch {
    @{status='notification_failed'; error=$_.Exception.Message; updated_at=(Get-Date).ToString('o')} |
        ConvertTo-Json | Set-Content -LiteralPath $statePath -Encoding UTF8
    throw
} finally {
    $mutex.ReleaseMutex()
    $mutex.Dispose()
}
