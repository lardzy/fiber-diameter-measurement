@echo off
call "C:\BuildTools\VC\Auxiliary\Build\vcvars64.bat"
if errorlevel 1 exit /b %errorlevel%
cd /d "C:\tmp\fdm-cu-live-ui-redesign-20261016"
cl /nologo /std:c++17 /EHsc /LD tests\native\microview_test_sdk.cpp /Fo:build\test-sdk\sdk.obj /Fe:build\test-sdk\MVAPI.dll /link user32.lib
exit /b %errorlevel%
