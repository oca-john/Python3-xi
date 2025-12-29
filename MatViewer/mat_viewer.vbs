Set WshShell = CreateObject("WScript.Shell")
WshShell.Run "cmd /c cd /d ""%~dp0"" & python mat_viewer.py", 0, False
