# Setup venv modo_train (Python 3.12 + torch ROCm 7.2.1) para RX 9070 XT.
# Correr desde la RAIZ del repo en PowerShell:
#   .\scripts\setup_modo_train.ps1
#   .\scripts\setup_modo_train.ps1 -VenvDir D:\vly_train   # otra ubicacion
#
# CRITICO: el venv va en una ruta SIN ESPACIOS (default C:\vly_train).
# El repo esta en "...\stats Vol" (con espacio) y el stack ROCm-Windows preview
# NO escapa espacios: MIOpen compila kernels HIP en runtime con clang y el -I a
# sus headers se trunca en el espacio -> "'type_traits' file not found" y el
# train muere con miopenStatusUnknownError. Por eso el venv NO puede vivir aca.
#
# Por que script y no pip a mano: las URLs del repo AMD usan sintaxis CMD (^ y
# &&) que PowerShell NO entiende. Aca van como arrays. Se llama python.exe
# directo (no se activa el venv).

param([string]$VenvDir = "C:\vly_train")

$ErrorActionPreference = "Stop"

if ($VenvDir -match "\s") {
    Write-Error "VenvDir '$VenvDir' tiene espacios. ROCm-Windows no los soporta. Usa una ruta sin espacios."
}

# Borrar venvs mal ubicados de intentos previos (dentro del repo con espacio)
foreach ($bad in @(".\modo_train", ".\data\dataset\modo_train")) {
    if (Test-Path $bad) {
        Write-Host "Borrando venv en ruta con espacio: $bad ..."
        Remove-Item -Recurse -Force $bad
    }
}

# Crear venv 3.12 en ruta sin espacios (las wheels son cp312; 3.13 NO sirve)
if (-not (Test-Path "$VenvDir\Scripts\python.exe")) {
    Write-Host "Creando venv en $VenvDir con Python 3.12 ..."
    py -3.12 -m venv $VenvDir
}
$py = "$VenvDir\Scripts\python.exe"

$ver = & $py --version
Write-Host "venv python: $ver"
if ($ver -notmatch "3\.12") {
    Write-Error "El venv no es Python 3.12 ($ver). Instala 3.12: winget install Python.Python.3.12"
}

& $py -m pip install --upgrade pip

# ROCm SDK
$rocm = @(
    "https://repo.radeon.com/rocm/windows/rocm-rel-7.2.1/rocm_sdk_core-7.2.1-py3-none-win_amd64.whl",
    "https://repo.radeon.com/rocm/windows/rocm-rel-7.2.1/rocm_sdk_devel-7.2.1-py3-none-win_amd64.whl",
    "https://repo.radeon.com/rocm/windows/rocm-rel-7.2.1/rocm_sdk_libraries_custom-7.2.1-py3-none-win_amd64.whl",
    "https://repo.radeon.com/rocm/windows/rocm-rel-7.2.1/rocm-7.2.1.tar.gz"
)
Write-Host "Instalando ROCm SDK ..."
& $py -m pip install --no-cache-dir $rocm

# torch + torchvision ROCm
$torch = @(
    "https://repo.radeon.com/rocm/windows/rocm-rel-7.2.1/torch-2.9.1%2Brocm7.2.1-cp312-cp312-win_amd64.whl",
    "https://repo.radeon.com/rocm/windows/rocm-rel-7.2.1/torchvision-0.24.1%2Brocm7.2.1-cp312-cp312-win_amd64.whl"
)
Write-Host "Instalando torch + torchvision ROCm ..."
& $py -m pip install --no-cache-dir $torch

# ultralytics (NO reinstala torch: 2.9.1 ya satisface el requisito)
Write-Host "Instalando ultralytics ..."
& $py -m pip install ultralytics

# Smoke test GPU
Write-Host "`n=== Smoke test GPU ==="
& $py -c "import torch; print('torch', torch.__version__); print('cuda', torch.cuda.is_available()); print('device', torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'NINGUNA')"

Write-Host "`nListo. Si cuda=True, disparar el train (python del venv + ruta del script):"
Write-Host "  & `"$py`" `"$PWD\scripts\train_ball.py`""
