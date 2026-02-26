@echo off
REM build_gpu.bat — Compileert de CUDA-kernel en bouwt azen-gpu.exe
REM Vereisten: nvcc (CUDA 12.8), Go 1.22+
REM Gebruik:   build_gpu.bat

setlocal

set CUDA_DIR=C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.8
set CUDA_BIN=%CUDA_DIR%\bin
set OUT_DLL=simulate.dll
set SRC_CU=cuda\simulate.cu

echo.
echo === Stap 1: CUDA-kernel compileren naar DLL ===
echo.

"%CUDA_BIN%\nvcc.exe" ^
    "%SRC_CU%" ^
    -shared ^
    -o "%OUT_DLL%" ^
    -arch=sm_89 ^
    -O3

if errorlevel 1 (
    echo [FOUT] CUDA compilatie mislukt.
    exit /b 1
)

echo [OK] %OUT_DLL% aangemaakt.

echo.
echo === Stap 2: Go-executable bouwen ===
echo.

go build -o azen-gpu.exe .

if errorlevel 1 (
    echo [FOUT] Go build mislukt.
    exit /b 1
)

echo [OK] azen-gpu.exe aangemaakt.
echo.
echo Gereed! Start met:  azen-gpu.exe
echo (simulate.dll moet in dezelfde map staan als azen-gpu.exe)

endlocal
