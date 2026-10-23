@echo off
cd /d "C:\tmp\fdm-brightness-release-20261023"
set PYTHONPATH=C:\tmp\fdm-brightness-release-20261023\src;C:\tmp\fdm-brightness-release-20261023\validation_vendor
set PYTHONUTF8=1
set PYTHONDONTWRITEBYTECODE=1
set PIP_NO_INDEX=1
set APPDATA=C:\tmp\fdm-brightness-release-20261023\build\isolated-config
set LOCALAPPDATA=C:\tmp\fdm-brightness-release-20261023\build\isolated-config
set TEMP=C:\tmp\fdm-brightness-release-20261023\build\temp
set TMP=C:\tmp\fdm-brightness-release-20261023\build\temp
"C:\Users\lishuyang\Downloads\fiber-diameter-measurement\.venv\Scripts\python.exe" scripts\build_windows_installer.py --reuse-onedir --exclude-area-models > build\repackage.log 2>&1
set FDM_RELEASE_EXIT=%errorlevel%
echo %FDM_RELEASE_EXIT% > build\repackage-exit.txt
exit /b %FDM_RELEASE_EXIT%
