$ErrorActionPreference = "Stop"
Set-StrictMode -Version Latest

$repoRoot = (Resolve-Path (Join-Path $PSScriptRoot "..")).Path
$portForwardProcess = $null
Push-Location $repoRoot

try {
    foreach ($tool in @("docker", "kubectl")) {
        if (-not (Get-Command $tool -ErrorAction SilentlyContinue)) {
            throw "$tool was not found. Start Docker Desktop and ensure its CLI tools are on PATH."
        }
    }

    & docker info *> $null
    if ($LASTEXITCODE -ne 0) {
        throw "Docker Desktop is not running. Start it and try again."
    }

    & kubectl get nodes
    if ($LASTEXITCODE -ne 0) {
        throw "Docker Desktop Kubernetes is not ready. Enable it in Docker Desktop Settings > Kubernetes."
    }

    if (-not (Test-Path (Join-Path $repoRoot ".env"))) {
        throw "The project .env is missing. Run scripts\start.ps1 once to initialize Docker Compose first."
    }

    & kubectl get secret ischemia-secrets -n ischemia *> $null
    if ($LASTEXITCODE -ne 0) {
        throw "Kubernetes secrets are not initialized. Follow the first-time setup in RUN_COMMANDS.txt."
    }

    Write-Host "`nChecking and running DVC training..."
    & docker compose run --rm trainer dvc repro
    if ($LASTEXITCODE -ne 0) {
        throw "DVC training failed."
    }

    Write-Host "`nBuilding and starting Docker Compose..."
    & docker compose up --build -d
    if ($LASTEXITCODE -ne 0) {
        throw "Docker Compose failed to start."
    }

    Write-Host "`nBuilding the Kubernetes image..."
    & docker build -t ischemia-predictor:local .
    if ($LASTEXITCODE -ne 0) {
        throw "Kubernetes image build failed."
    }

    Write-Host "`nApplying Kubernetes configuration..."
    & kubectl apply -f .\k8s\deployment.yaml
    if ($LASTEXITCODE -ne 0) {
        throw "Could not apply the Kubernetes configuration."
    }

    & kubectl rollout restart deployment/web deployment/ml deployment/dl deployment/qml -n ischemia
    if ($LASTEXITCODE -ne 0) {
        throw "Could not restart the Kubernetes app deployments."
    }

    foreach ($deployment in @("web", "ml", "dl", "qml")) {
        & kubectl rollout status "deployment/$deployment" -n ischemia --timeout=180s
        if ($LASTEXITCODE -ne 0) {
            throw "Kubernetes deployment '$deployment' did not become ready."
        }
    }

    $kubernetesPort = 8081
    $listener = $null
    while ($kubernetesPort -le 8090) {
        try {
            $listener = New-Object System.Net.Sockets.TcpListener(
                [System.Net.IPAddress]::Loopback, $kubernetesPort
            )
            $listener.Start()
            $listener.Stop()
            break
        }
        catch {
            if ($listener) {
                $listener.Stop()
            }
            $listener = $null
            $kubernetesPort++
        }
    }
    if ($kubernetesPort -gt 8090) {
        throw "No available localhost port between 8081 and 8090 for Kubernetes."
    }

    $forwardCommand = "kubectl port-forward service/web ${kubernetesPort}:80 -n ischemia"
    $portForwardProcess = Start-Process -FilePath "powershell.exe" `
        -ArgumentList @("-NoExit", "-Command", $forwardCommand) -PassThru

    $kubernetesReady = $false
    for ($attempt = 0; $attempt -lt 20; $attempt++) {
        try {
            $page = Invoke-WebRequest -Uri "http://localhost:$kubernetesPort/" -UseBasicParsing -TimeoutSec 3
            if ($page.StatusCode -eq 200 -and $page.Content.Contains("loginView")) {
                $kubernetesReady = $true
                break
            }
        }
        catch {
            Start-Sleep -Seconds 1
        }
    }
    if (-not $kubernetesReady) {
        Stop-Process -Id $portForwardProcess.Id -ErrorAction SilentlyContinue
        throw "Kubernetes port-forward did not serve the updated UI. Check its PowerShell window."
    }

    Write-Host "`nCompose app:     http://localhost:5000"
    Write-Host "Compose MLflow:  http://localhost:5001"
    Write-Host "Kubernetes app:  http://localhost:$kubernetesPort"
    Write-Host "Kubernetes pods and service connections:"
    & kubectl get pods,services,endpoints -n ischemia

    Start-Process "http://localhost:5000"
    Start-Process "http://localhost:$kubernetesPort"

    Read-Host "Both apps are open. Press Enter here when you are finished; this also stops Kubernetes port-forwarding"
    if (Get-Process -Id $portForwardProcess.Id -ErrorAction SilentlyContinue) {
        Stop-Process -Id $portForwardProcess.Id
    }
}
finally {
    Pop-Location
}
