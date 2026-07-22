$ErrorActionPreference = "Continue"

$logDirectory = "experiment_logs\batchnorm_robust"

New-Item `
    -ItemType Directory `
    -Force `
    -Path $logDirectory |
    Out-Null


$experiments = @(

    @{
        Name = "batchnorm_clean_robust"
        Strategy = "robust"
        Attack = "none"
        PoisonFraction = "0.0"
        GradientClip = ""
    },

    @{
        Name = "batchnorm_label_flip_10pct_robust"
        Strategy = "robust"
        Attack = "label_flip"
        PoisonFraction = "0.10"
        GradientClip = ""
    },

    @{
        Name = "batchnorm_gradclip_clean_robust"
        Strategy = "robust"
        Attack = "none"
        PoisonFraction = "0.0"
        GradientClip = "1.0"
    },

    @{
        Name = "batchnorm_gradclip_label_flip_10pct_robust"
        Strategy = "robust"
        Attack = "label_flip"
        PoisonFraction = "0.10"
        GradientClip = "1.0"
    }

)


foreach ($experiment in $experiments) {

    $name = $experiment.Name
    $strategy = $experiment.Strategy
    $attack = $experiment.Attack
    $poisonFraction = $experiment.PoisonFraction
    $gradientClip = $experiment.GradientClip

    $logFile = Join-Path `
        $logDirectory `
        "$name.log"

    Write-Host ""
    Write-Host "================================================"
    Write-Host "Starting: $name"
    Write-Host "Log file: $logFile"
    Write-Host "================================================"

    if ($gradientClip -eq "") {

        python -u -m tests.test_50_client_simulation `
            --experiment-name $name `
            --strategy $strategy `
            --attack $attack `
            --poison-fraction $poisonFraction `
            2>&1 |
            Tee-Object -FilePath $logFile

    }
    else {

        python -u -m tests.test_50_client_simulation `
            --experiment-name $name `
            --strategy $strategy `
            --attack $attack `
            --poison-fraction $poisonFraction `
            --gradient-clip $gradientClip `
            2>&1 |
            Tee-Object -FilePath $logFile

    }

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
Write-Host "ALL BATCHNORM ROBUST EXPERIMENTS COMPLETED"
Write-Host "Logs saved in: $logDirectory"
Write-Host "================================================"