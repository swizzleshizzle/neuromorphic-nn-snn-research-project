# Start chain085.ps1 detached from the ssh session (Win32_Process.Create), so a dropped
# connection never takes the run with it. Output goes to C:\Users\mlgbr\chain085.out.
$repo = "C:\Users\mlgbr\Desktop\Projects\neuromorphic-nn-snn-research-project"
$cmd = 'cmd /c "powershell -NoProfile -ExecutionPolicy Bypass -File ' + $repo + '\experiments\085_outward_walks\chain085.ps1 >> C:\Users\mlgbr\chain085.out 2>&1"'
$r = Invoke-CimMethod -ClassName Win32_Process -MethodName Create -Arguments @{CommandLine=$cmd; CurrentDirectory=$repo}
"rv=$($r.ReturnValue) pid=$($r.ProcessId)"
