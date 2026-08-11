[CmdletBinding()]
param(
    [Parameter(Mandatory = $true)]
    [ValidateSet("no-traffic", "traffic")]
    [string]$Mode,

    [ValidateSet("smoke", "full", "matrix")]
    [string]$Profile = "smoke",

    [ValidateSet("cpu", "cuda")]
    [string]$Device = "cpu",

    [ValidateRange(0, [int]::MaxValue)]
    [int]$Seed = 0,

    [ValidateRange(0, 20)]
    [int[]]$Seeds = @(0, 1, 2),

    [ValidateRange(0.000001, 1.0)]
    [double[]]$TrafficDensities = @(0.05, 0.10),

    [switch]$Video,

    [switch]$DryRun
)

Set-StrictMode -Version Latest
$ErrorActionPreference = "Stop"

function Join-OverrideValues {
    param([Parameter(Mandatory = $true)][object[]]$Values)

    return ($Values | ForEach-Object { $_.ToString([Globalization.CultureInfo]::InvariantCulture) }) -join ","
}

$repositoryRoot = (Resolve-Path (Join-Path $PSScriptRoot "..")).Path
$requiredAssets = @(
    (Join-Path $repositoryRoot "checkpoints\DP-Origin\args.json"),
    (Join-Path $repositoryRoot "checkpoints\DP-Origin\model.pth"),
    (Join-Path $repositoryRoot "third_party\metadrive")
)
$missingAssets = @($requiredAssets | Where-Object { -not (Test-Path -LiteralPath $_) })
if ($missingAssets.Count -gt 0) {
    throw "Required evaluation assets are missing: $($missingAssets -join ', ')"
}
if (-not (Get-Command uv -ErrorAction SilentlyContinue)) {
    throw "uv is not on PATH. Install uv and run 'uv sync --all-groups' before evaluating."
}

$uvArguments = [System.Collections.Generic.List[string]]::new()
$uvArguments.Add("run")
$uvArguments.Add("python")
$uvArguments.Add("scripts/evaluate.py")
if ($Mode -eq "traffic") {
    $uvArguments.Add("--config-name")
    $uvArguments.Add("evaluation/traffic")
}
if ($Profile -eq "matrix") {
    $uvArguments.Add("--multirun")
}

$videoEnabled = $Video.IsPresent.ToString().ToLowerInvariant()
$overrides = [System.Collections.Generic.List[string]]::new()
$overrides.Add("video.enabled=$videoEnabled")
$overrides.Add("model.device=$Device")

if ($Mode -eq "no-traffic") {
    switch ($Profile) {
        "smoke" {
            $overrides.Add("model.seed=$Seed")
            $overrides.Add("evaluation.evaluated_horizon_steps=20")
            $overrides.Add("env.horizon=20")
            $overrides.Add("scenarios=[{name:straight,map:S,seed:$Seed}]")
        }
        "full" {
            $overrides.Add("model.seed=$Seed")
        }
        "matrix" {
            $overrides.Add("model.seed=$(Join-OverrideValues $Seeds)")
        }
    }
}
else {
    switch ($Profile) {
        "smoke" {
            $overrides.Add("seed=$Seed")
            $overrides.Add("evaluation.evaluated_horizon_steps=100")
            $overrides.Add("env.horizon=120")
        }
        "full" {
            $overrides.Add("seed=$Seed")
        }
        "matrix" {
            $overrides.Add("seed=$(Join-OverrideValues $Seeds)")
            $overrides.Add("env.traffic_density=$(Join-OverrideValues $TrafficDensities)")
        }
    }
}

foreach ($override in $overrides) {
    $uvArguments.Add($override)
}

Push-Location $repositoryRoot
try {
    Write-Host "Running from $repositoryRoot"
    Write-Host "uv $($uvArguments -join ' ')"
    if ($DryRun) {
        return
    }
    & uv @uvArguments
    if ($LASTEXITCODE -ne 0) {
        throw "Evaluation failed with uv exit code $LASTEXITCODE."
    }
}
finally {
    Pop-Location
}
