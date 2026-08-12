[CmdletBinding()]
param(
    [Parameter(Mandatory = $true)]
    [ValidateSet("no-traffic", "traffic")]
    [string]$Mode,

    [ValidateSet("smoke", "full", "matrix")]
    [string]$Profile = "smoke",

    [ValidateSet("auto", "cpu", "cuda")]
    [string]$Accelerator = "auto",

    [ValidateSet("cpu", "cuda")]
    [string]$Extra = "cpu",

    [ValidateSet("auto", "32-true", "16-mixed", "bf16-mixed")]
    [string]$Precision = "auto",

    [ValidateSet("dpm10", "ddim5", "ddim5_project_noise")]
    [string]$Sampler = "dpm10",

    [ValidateSet("none", "orthogonal_reference")]
    [string]$Guidance = "none",

    [ValidateRange(-1.0, 1.0)]
    [double]$LateralScale = 0.0,

    [ValidateRange(-1.0, 1.0)]
    [double]$LongitudinalScale = 0.0,

    [ValidateRange(0, [int]::MaxValue)]
    [int]$RuntimeSeed = 0,

    [ValidateRange(0, 20)]
    [int[]]$RuntimeSeeds = @(0, 1, 2),

    [ValidateRange(0, [int]::MaxValue)]
    [int]$ScenarioSeed = 0,

    [ValidateRange(0.000001, 1.0)]
    [double[]]$TrafficDensities = @(0.05, 0.10),

    [ValidateSet("auto", "serial", "parallel")]
    [string]$ExecutionMode = "auto",

    [ValidateRange(1, [int]::MaxValue)]
    [int]$TorchThreadsPerWorker = 8,

    [switch]$Video,

    [switch]$DryRun
)

Set-StrictMode -Version Latest
$ErrorActionPreference = "Stop"

if ($Guidance -eq "orthogonal_reference" -and $Sampler -ne "ddim5") {
    throw "orthogonal_reference guidance requires -Sampler ddim5."
}
if ($Guidance -eq "none" -and ($LateralScale -ne 0.0 -or $LongitudinalScale -ne 0.0)) {
    throw "Non-zero guidance scales require -Guidance orthogonal_reference."
}
if ($Accelerator -eq "cuda" -and $Extra -ne "cuda") {
    throw "-Accelerator cuda requires -Extra cuda."
}
if ($Accelerator -eq "cpu" -and $Extra -ne "cpu") {
    throw "-Accelerator cpu requires -Extra cpu."
}

$resolvedExecutionMode = $ExecutionMode
if ($resolvedExecutionMode -eq "auto") {
    $resolvedExecutionMode = if ($Mode -eq "traffic" -and $Profile -eq "matrix") {
        "parallel"
    }
    else {
        "serial"
    }
}
if ($resolvedExecutionMode -eq "parallel" -and ($Mode -ne "traffic" -or $Profile -ne "matrix")) {
    throw "Parallel execution is only supported for the traffic matrix profile."
}
if ($resolvedExecutionMode -eq "parallel" -and $Video.IsPresent) {
    throw "Parallel execution requires video to remain disabled."
}
if ($resolvedExecutionMode -eq "parallel" -and $Accelerator -eq "cpu") {
    $threadBudget = 2 * $TorchThreadsPerWorker
    if ($threadBudget -gt [Environment]::ProcessorCount) {
        throw "Parallel CPU thread budget $threadBudget exceeds $([Environment]::ProcessorCount) logical CPUs."
    }
}

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
$uvArguments.Add("--extra")
$uvArguments.Add($Extra)
$uvArguments.Add("python")
$uvArguments.Add("scripts/evaluate.py")
if ($Mode -eq "traffic" -and $Profile -eq "matrix") {
    $uvArguments.Add("--config-name")
    $uvArguments.Add("evaluation/traffic_matrix")
}
elseif ($Mode -eq "traffic") {
    $uvArguments.Add("--config-name")
    $uvArguments.Add("evaluation/traffic")
}
if ($Profile -eq "matrix") {
    $uvArguments.Add("--multirun")
}

$videoEnabled = $Video.IsPresent.ToString().ToLowerInvariant()
$overrides = [System.Collections.Generic.List[string]]::new()
$overrides.Add("video.enabled=$videoEnabled")
$overrides.Add("runtime.accelerator=$Accelerator")
$overrides.Add("runtime.precision=$Precision")
$overrides.Add("sampler=$Sampler")
$overrides.Add("guidance=$Guidance")
$overrides.Add("evaluation.execution.torch_threads_per_worker=$TorchThreadsPerWorker")
if ($resolvedExecutionMode -eq "parallel") {
    $overrides.Add("evaluation.execution.mode=parallel")
    $overrides.Add("evaluation.execution.launcher=joblib")
    $overrides.Add("evaluation.execution.worker_count=2")
    $overrides.Add("evaluation.execution.deterministic=true")
    $overrides.Add("hydra/launcher=joblib")
    $overrides.Add("hydra.launcher.n_jobs=2")
    $overrides.Add("hydra.launcher.backend=loky")
    $overrides.Add("hydra.launcher.prefer=processes")
    $overrides.Add("hydra.launcher.batch_size=1")
}
else {
    $overrides.Add("evaluation.execution.mode=serial")
    $overrides.Add("evaluation.execution.launcher=basic")
    $overrides.Add("evaluation.execution.worker_count=1")
    $overrides.Add("hydra/launcher=basic")
}
if ($Guidance -eq "orthogonal_reference") {
    $lateralText = $LateralScale.ToString([Globalization.CultureInfo]::InvariantCulture)
    $longitudinalText = $LongitudinalScale.ToString([Globalization.CultureInfo]::InvariantCulture)
    $overrides.Add("guidance.lateral_scale=$lateralText")
    $overrides.Add("guidance.longitudinal_scale=$longitudinalText")
}

if ($Mode -eq "no-traffic") {
    switch ($Profile) {
        "smoke" {
            $overrides.Add("runtime.seed=$RuntimeSeed")
            $overrides.Add("evaluation.evaluated_horizon_steps=20")
            $overrides.Add("env.horizon=20")
            $overrides.Add("scenarios=[{name:straight,map:S,seed:$ScenarioSeed}]")
        }
        "full" {
            $overrides.Add("runtime.seed=$RuntimeSeed")
        }
        "matrix" {
            $overrides.Add("runtime.seed=$(Join-OverrideValues $RuntimeSeeds)")
        }
    }
}
else {
    switch ($Profile) {
        "smoke" {
            $overrides.Add("runtime.seed=$RuntimeSeed")
            $overrides.Add("evaluation.evaluated_horizon_steps=100")
            $overrides.Add("env.horizon=120")
        }
        "full" {
            $overrides.Add("runtime.seed=$RuntimeSeed")
        }
        "matrix" {
            $overrides.Add("runtime.seed=$(Join-OverrideValues $RuntimeSeeds)")
            $overrides.Add("env.traffic_density=$(Join-OverrideValues $TrafficDensities)")
            $overrides.Add("evaluation.matrix.seeds=[$(Join-OverrideValues $RuntimeSeeds)]")
            $overrides.Add(
                "evaluation.matrix.traffic_densities=[$(Join-OverrideValues $TrafficDensities)]"
            )
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
    $cudaAvailable = $false
    if ($resolvedExecutionMode -eq "parallel" -and $Accelerator -in @("auto", "cuda")) {
        $cudaAvailableText = & uv run --extra $Extra python -c "import torch; print(str(torch.cuda.is_available()).lower())"
        if ($LASTEXITCODE -ne 0) {
            throw "Failed to resolve CUDA availability before parallel evaluation."
        }
        $cudaAvailable = $cudaAvailableText.Trim() -eq "true"
    }
    if ($Accelerator -eq "cuda" -and -not $cudaAvailable) {
        throw "CUDA was requested but the active uv environment cannot use it."
    }
    if ($cudaAvailable) {
        $env:CUBLAS_WORKSPACE_CONFIG = ":4096:8"
        $preflightDir = "outputs/cuda_preflight/$([DateTime]::Now.ToString('yyyy-MM-dd/HH-mm-ss'))"
        $preflightArguments = @(
            "run", "--extra", $Extra, "python", "scripts/evaluate.py", "--multirun",
            "hydra/launcher=joblib", "hydra.launcher.n_jobs=2",
            "hydra.launcher.backend=loky", "hydra.launcher.prefer=processes",
            "runtime.accelerator=cuda", "runtime.precision=$Precision", "runtime.seed=0,1",
            "video.enabled=false", "evaluation.evaluated_horizon_steps=5", "env.horizon=5",
            "evaluation.execution.mode=parallel", "evaluation.execution.launcher=joblib",
            "evaluation.execution.worker_count=2",
            "evaluation.execution.torch_threads_per_worker=$TorchThreadsPerWorker",
            "evaluation.execution.deterministic=true",
            "scenarios=[{name:cuda_preflight,map:S,seed:0}]",
            "hydra.sweep.dir=$preflightDir"
        )
        Write-Host "Running two-worker CUDA memory preflight"
        & uv @preflightArguments
        if ($LASTEXITCODE -ne 0) {
            throw "CUDA parallel memory preflight failed."
        }
    }
    & uv @uvArguments
    if ($LASTEXITCODE -ne 0) {
        throw "Evaluation failed with uv exit code $LASTEXITCODE."
    }
}
finally {
    Pop-Location
}
