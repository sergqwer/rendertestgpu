@echo off
setlocal

set VSTOOLS=C:\Program Files\Microsoft Visual Studio\18\Community\VC\Tools\MSVC\14.16.27023\bin\HostX64\x64

echo Extracting Vulkan exports...
"%VSTOOLS%\dumpbin.exe" /exports C:\Windows\System32\vulkan-1.dll > vulkan_exports.txt

echo Creating DEF file...
echo LIBRARY vulkan-1 > vulkan-1.def
echo EXPORTS >> vulkan-1.def

for /f "tokens=4" %%a in ('findstr /R "^  *[0-9][0-9]*  *[0-9A-Fa-f]" vulkan_exports.txt') do (
    echo %%a >> vulkan-1.def
)

echo Creating LIB file...
"%VSTOOLS%\lib.exe" /def:vulkan-1.def /out:vulkan-1.lib /machine:x64

echo Done!
dir vulkan-1.*
