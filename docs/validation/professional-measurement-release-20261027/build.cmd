@echo off
call "C:\BuildTools\VC\Auxiliary\Build\vcvars64.bat"
if errorlevel 1 exit /b %errorlevel%
cd /d "C:\tmp\fdm-professional-measurement-release-20261027"
set PYTHONPATH=C:\tmp\fdm-professional-measurement-release-20261027\src;C:\tmp\fdm-professional-measurement-release-20261027\validation_vendor
set PYTHONUTF8=1
set PYTHONDONTWRITEBYTECODE=1
set PIP_NO_INDEX=1
set APPDATA=C:\tmp\fdm-professional-measurement-release-20261027\build\isolated-config
set LOCALAPPDATA=C:\tmp\fdm-professional-measurement-release-20261027\build\isolated-config
set TEMP=C:\tmp\fdm-professional-measurement-release-20261027\build\temp
set TMP=C:\tmp\fdm-professional-measurement-release-20261027\build\temp
"C:\Users\lishuyang\Downloads\fiber-diameter-measurement\.venv\Scripts\python.exe" setup.py build_ext --inplace
if errorlevel 1 exit /b %errorlevel%
"C:\Users\lishuyang\Downloads\fiber-diameter-measurement\.venv\Scripts\python.exe" scripts\build_windows_installer.py --exclude-area-models
exit /b %errorlevel%
