@echo off
setlocal
call "%~dp0build_windows_installer.bat" --both-area-variants %*
exit /b %ERRORLEVEL%
