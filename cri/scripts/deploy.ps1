Write-Host "========================================"
Write-Host "Starting CRI Chatbot Deployment"
Write-Host "========================================"

Set-Location "D:\cnc"

Write-Host "Pulling latest code..."
git pull origin main

if ($LASTEXITCODE -ne 0) {
    Write-Error "Git pull failed."
    exit 1
}

Write-Host "Restarting CRIChatbot service..."
Restart-Service -Name "CRIChatbot" -Force

Start-Sleep -Seconds 10

try {
    $response = Invoke-WebRequest `
        -Uri "http://10.225.30.26:8085/docs" `
        -UseBasicParsing `
        -TimeoutSec 20

    if ($response.StatusCode -eq 200) {
        Write-Host "Health check passed."
    }
    else {
        throw "Unexpected status code $($response.StatusCode)"
    }
}
catch {
    Write-Error "Health check failed."
    exit 1
}

Write-Host "Deployment completed successfully."