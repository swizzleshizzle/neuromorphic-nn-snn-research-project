# One status line for the VPS watcher: verdict present | python processes | chain processes |
# training records | search records | Gate H records | last chain log line.
$o = "D:\gpu083\repo\experiments\085_outward_walks\outputs"
$v = Test-Path "D:\gpu083\verdict085.txt"
$t = @(Get-ChildItem $o -Filter "exp085_[CO]_u24000_s*.json" -ErrorAction SilentlyContinue).Count
$s = @(Get-ChildItem $o -Filter "exp085_J4V-*.json" -ErrorAction SilentlyContinue).Count
$h = @(Get-ChildItem $o -Filter "exp085_H*.json" -ErrorAction SilentlyContinue).Count
$n = @(Get-Process | Where-Object { $_.ProcessName -match "^python" }).Count
$c = @(Get-CimInstance Win32_Process | Where-Object { $_.CommandLine -match "chain085" -and $_.Name -match "powershell" }).Count
$last = ""
if (Test-Path "D:\gpu083\chain085.log") { $last = (Get-Content "D:\gpu083\chain085.log" -Tail 1) }
"$v|py $n|ch $c|T $t|S $s|H $h|$last"
