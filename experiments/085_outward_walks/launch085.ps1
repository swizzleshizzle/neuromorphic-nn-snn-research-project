# Start chain085.ps1 detached from the ssh session (Win32_Process.Create), so a dropped
# connection never takes the run with it. Output goes to D:\gpu083\chain085.out.
$cmd = 'cmd /c "powershell -NoProfile -ExecutionPolicy Bypass -File D:\gpu083\repo\experiments\085_outward_walks\chain085.ps1 >> D:\gpu083\chain085.out 2>&1"'
$r = Invoke-CimMethod -ClassName Win32_Process -MethodName Create -Arguments @{CommandLine=$cmd; CurrentDirectory='D:\gpu083\repo'}
"rv=$($r.ReturnValue) pid=$($r.ProcessId)"
