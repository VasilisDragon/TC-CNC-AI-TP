Param()

function Test-Command {
    param(
        [Parameter(Mandatory = $true)][string]$Name
    )
    return [bool](Get-Command $Name -ErrorAction SilentlyContinue)
}

Write-Host "AIToolpathGenerator environment verification" -ForegroundColor Cyan
Write-Host ""

$cmakeOk = Test-Command -Name "cmake"
if ($cmakeOk) {
    $cmakeVersion = (& cmake --version).Split([Environment]::NewLine)[0]
    Write-Host "[OK] CMake: $cmakeVersion"
} else {
    Write-Warning "CMake not found. Install from https://cmake.org/download/ or via Visual Studio Installer."
}

$ninjaOk = Test-Command -Name "ninja"
if ($ninjaOk) {
    $ninjaVersion = (& ninja --version)
    Write-Host "[OK] Ninja: $ninjaVersion"
} else {
    Write-Warning "Ninja not found. Install via https://github.com/ninja-build/ninja/releases and ensure it is on PATH."
}

$clOk = $false
try {
    $clOutput = & cl.exe 2>$null
    if ($LASTEXITCODE -eq 0 -or $LASTEXITCODE -eq 2) {
        $clOk = $true
    }
} catch {
    $clOk = $false
}

if ($clOk) {
    Write-Host "[OK] MSVC (cl.exe) detected."
} else {
    Write-Warning "MSVC compiler not detected. Launch 'x64 Native Tools Command Prompt for VS 2022' before building."
}

$qtHint = $env:Qt6_DIR
if (-not $qtHint) { $qtHint = $env:Qt6Core_DIR }
if (-not $qtHint) { $qtHint = $env:QT_DIR }
if ($qtHint) {
    Write-Host "[OK] Qt hint detected at '$qtHint'."
} else {
    Write-Warning "Qt6 CMake package not located. Set Qt6_DIR (e.g. C:\Qt\6.5.3\msvc2019_64\lib\cmake\Qt6)."
}

function Test-OptionalBackend {
    param(
        [string]$EnvVar,
        [string]$Display,
        [string]$Toggle
    )
    $value = Get-Item "Env:$EnvVar" -ErrorAction SilentlyContinue
    if ($null -ne $value) {
        Write-Host "[OK] $Display detected via $EnvVar."
    } else {
        Write-Host "[INFO] $Display not detected. Enable with -D$Toggle=ON only when the SDK is installed."
    }
}

Test-OptionalBackend -EnvVar "TORCH_DIR" -Display "LibTorch" -Toggle "WITH_TORCH"
Test-OptionalBackend -EnvVar "onnxruntime_DIR" -Display "ONNX Runtime" -Toggle "WITH_ONNXRUNTIME"
Test-OptionalBackend -EnvVar "OCL_DIR" -Display "OpenCAMLib" -Toggle "WITH_OCL"

Write-Host ""
if ($cmakeOk -and $ninjaOk -and $clOk) {
    Write-Host "Environment looks good. Run: cmake --preset ninja-release" -ForegroundColor Green
} else {
    Write-Warning "Resolve the warnings above before configuring the project."
}
