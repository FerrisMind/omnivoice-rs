$ErrorActionPreference = "Stop"
Set-Location "C:\Users\PC\Documents\Default_Project\omnivoice-rs"
while ($true) {
    python tools/bench_one_at_a_time.py
    if ($LASTEXITCODE -ne 0) {
        Write-Host "Stopped on error (exit $LASTEXITCODE)" -ForegroundColor Red
        exit $LASTEXITCODE
    }
    $json = Get-Content "artifacts/bench_scenarios_gguf_q4km_cpu/results.json" -Raw | ConvertFrom-Json
    $count = @($json.runs.rust_cpu_auto.rows | Where-Object { $_.ok }).Count
    Write-Host "Verified: $count/15 scenarios in results.json"
    if ($count -ge 15) {
        Write-Host "All 15 scenarios complete."
        exit 0
    }
}
