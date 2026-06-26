@echo off
call "C:\Program Files (x86)\Microsoft Visual Studio\18\BuildTools\VC\Auxiliary\Build\vcvars64.bat"
rmdir /s /q build 2>nul
cmake -S . -B build -G "Ninja" -DCMAKE_BUILD_TYPE=Release
if %errorlevel% neq 0 (
    echo CMake configure FAILED
    exit /b 1
)
echo CMake configure OK
cmake --build build --parallel
if %errorlevel% neq 0 (
    echo Build FAILED
    exit /b 1
)
echo Build SUCCESS
