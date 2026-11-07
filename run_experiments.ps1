# ==============================================================================
# IP-GCA Copilot: Script Unificado de Reproducibilidad Experimental
# Permite a cualquier revisor o jurado de tesis reproducir todos los resultados,
# pruebas unitarias, benchmarks de ablación A-H y figuras de publicación.
# ==============================================================================

param(
    [switch]$SkipInstall = $false,
    [switch]$SkipTests = $false,
    [switch]$RunBenchmark = $true,
    [switch]$RunLoraExperiment = $true,
    [switch]$GenerateFigures = $true,
    [switch]$RunDemo = $false,
    [switch]$RunDesktopApp = $false
)

$ErrorActionPreference = "Stop"

Write-Host "`n==============================================================================" -ForegroundColor Cyan
Write-Host "       IP-GCA: Implicit Preference-Gated Copilot Architecture" -ForegroundColor Cyan
Write-Host "                Protocolo de Reproducibilidad de Tesis" -ForegroundColor Cyan
Write-Host "==============================================================================`n" -ForegroundColor Cyan

# 1. Verificación de Python y Entorno Virtual
$PythonExe = ".\.venv\Scripts\python.exe"
$PipExe = ".\.venv\Scripts\pip.exe"
$PytestExe = ".\.venv\Scripts\pytest.exe"

if (-not (Test-Path $PythonExe)) {
    Write-Host "[1/5] Entorno virtual no detectado. Creando .venv..." -ForegroundColor Yellow
    python -m venv .venv
}

# 2. Instalación de Dependencias
if (-not $SkipInstall) {
    Write-Host "[2/5] Verificando e instalando dependencias (requirements.txt)..." -ForegroundColor Yellow
    & $PipExe install -r requirements.txt --quiet
    Write-Host "  -> Dependencias listas." -ForegroundColor Green
} else {
    Write-Host "[2/5] Omitiendo instalación de dependencias (-SkipInstall activo)." -ForegroundColor DarkGray
}

# 3. Pruebas Unitarias de Validación de la Arquitectura
if (-not $SkipTests) {
    Write-Host "`n[3/5] Ejecutando suite de pruebas unitarias críticas..." -ForegroundColor Yellow
    & $PytestExe tests/test_fsm.py tests/test_adaptation.py tests/test_router.py tests/test_metrics.py -v
    if ($LASTEXITCODE -ne 0) {
        Write-Host "  -> ERROR: Una o más pruebas fallaron." -ForegroundColor Red
        exit $LASTEXITCODE
    }
    Write-Host "  -> Todas las pruebas de validación pasaron con éxito." -ForegroundColor Green
} else {
    Write-Host "`n[3/5] Omitiendo pruebas unitarias (-SkipTests activo)." -ForegroundColor DarkGray
}

# 4. Ejecución del Benchmark de Ablación A-H
if ($RunBenchmark) {
    Write-Host "`n[4/6] Ejecutando Matriz de Ablaciones A-H sobre 28 muestras de benchmark..." -ForegroundColor Yellow
    & $PythonExe src/run_ablation_benchmark.py
    if ($LASTEXITCODE -ne 0) {
        Write-Host "  -> ERROR al ejecutar el benchmark de ablación." -ForegroundColor Red
        exit $LASTEXITCODE
    }
    Write-Host "  -> Benchmark completado. Resultados en results/ablation_results.json" -ForegroundColor Green
}

# 5. Ejecución del Experimento de Adaptación LoRA (Offline Fine-Tuning)
if ($RunLoraExperiment) {
    Write-Host "`n[5/6] Ejecutando Experimento de Adaptación LoRA sobre 32 deltas y 16 pruebas held-out..." -ForegroundColor Yellow
    & $PythonExe src/experiments/evaluate_lora_adaptation.py
    if ($LASTEXITCODE -ne 0) {
        Write-Host "  -> ERROR al ejecutar experimento LoRA." -ForegroundColor Red
        exit $LASTEXITCODE
    }
    Write-Host "  -> Experimento LoRA completado. Tabla en results/table_lora_adaptation.tex" -ForegroundColor Green
}

# 6. Generación de Figuras de Publicación y Tablas LaTeX
if ($GenerateFigures) {
    Write-Host "`n[6/6] Generando figuras de alta resolución y tablas LaTeX para la tesis..." -ForegroundColor Yellow
    & $PythonExe src/generate_figures.py
    if ($LASTEXITCODE -ne 0) {
        Write-Host "  -> ERROR al generar figuras." -ForegroundColor Red
        exit $LASTEXITCODE
    }
    Write-Host "  -> Figuras guardadas en: results/figures/" -ForegroundColor Green
    Write-Host "  -> Tabla LaTeX guardada en: results/table_ablation_results.tex" -ForegroundColor Green
}

# 7. Demostración interactiva de escritorio (PyWebView)
if ($RunDesktopApp) {
    Write-Host "`n[Desktop App] Lanzando interfaz gráfica de escritorio (PyWebView)..." -ForegroundColor Magenta
    & $PythonExe src/ui/desktop_app.py
}

# 8. Demostración interactiva en consola opcional
if ($RunDemo) {
    Write-Host "`n[Extra] Lanzando demostración interactiva HITL en consola..." -ForegroundColor Magenta
    & $PythonExe src/copilot_demo.py
}

Write-Host "`n==============================================================================" -ForegroundColor Green
Write-Host "  Reproducibilidad finalizada exitosamente." -ForegroundColor Green
Write-Host "  Todos los artefactos requeridos para la tesis se encuentran generados." -ForegroundColor Green
Write-Host "==============================================================================`n" -ForegroundColor Green
