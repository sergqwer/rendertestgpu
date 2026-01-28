@echo off
set "MSBUILD=C:\Program Files\Microsoft Visual Studio\18\Community\MSBuild\Current\Bin\MSBuild.exe"
"%MSBUILD%" "%~dp0rendertestgpu.vcxproj" /p:Configuration=Release /p:Platform=x64 /t:Rebuild /nologo /v:minimal
