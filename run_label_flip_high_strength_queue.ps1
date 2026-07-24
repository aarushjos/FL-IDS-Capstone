$ErrorActionPreference = "Continue"

$logDirectory = "experiment_logs\label_flip_high_strength"

New-Item `
    -ItemType Directory `
    -Force `
    -Path $logDirectory |
    Out-Null


$experiments = @(

    @{
        Name = "batchnorm_label_flip_40pct_fedavg"
        Strategy = "fedavg"
        PoisonFraction = "0.40"
    },

    @{
        Name = "batchnorm_label_flip_40pct_robust"
        Strategy = "robust"
        PoisonFraction = "0.40"
    },

    @{
        Name = "batchnorm_label_flip_50pct_fedavg"
        Strategy = "fedavg"
        PoisonFraction = "0.50"
    },

    @{
        Name = "batchnorm_label_flip_50pct_robust"
        Strategy = "robust"
        PoisonFraction = "0.50"
    },

    @{
        Name = "batchnorm_label_flip_60pct_fedavg"
        Strategy = "fedavg"
        PoisonFraction = "0.60"
    },

    @{
        Name = "batchnorm_label_flip_60pct_robust"
        Strategy = "robust"
        PoisonFraction = "0.60"
    }

)


foreach ($experiment in $experiments) {

    $name = $experiment.Name
    $strategy = $experiment.Strategy
    $poisonFraction = $experiment.PoisonFraction

    $logFile = Join-Path `
        $logDirectory `
        "$name.log"

    Write-Host ""
    Write-Host "================================================"
    Write-Host "Starting: $name"
    Write-Host "Strategy: $strategy"
    Write-Host "Attack: label_flip"
    Write-Host "Poison fraction: $poisonFraction"
    Write-Host "Log file: $logFile"
    Write-Host "================================================"

    python -u -m tests.test_50_client_simulation `
        --experiment-name $name `
        --strategy $strategy `
        --attack label_flip `
        --poison-fraction $poisonFraction `
        2>&1 |
        Tee-Object -FilePath $logFile

    $exitCode = $LASTEXITCODE

    if ($exitCode -ne 0) {

        Write-Host ""
        Write-Host "$name failed with exit code $exitCode."
        Write-Host "Queue stopped."

        exit $exitCode
    }

    Write-Host ""
    Write-Host "$name completed successfully."

    Get-Process `
        raylet, gcs_server `
        -ErrorAction SilentlyContinue |
        Stop-Process -Force

    Start-Sleep -Seconds 10
}


Write-Host ""
Write-Host "================================================"
Write-Host "ALL HIGH-STRENGTH LABEL-FLIP RUNS COMPLETED"
Write-Host "Logs saved in: $logDirectory"
Write-Host "================================================"