# One status line for a VPS watcher: verdict present | python processes | Gate P, E records |
# Gate M record | Gate T records | last chain log line.
$o = "D:\gpu083\repo\experiments\084_gpu_port\outputs"
$v = Test-Path "D:\gpu083\verdict084.txt"
$p = @(Get-ChildItem $o -Filter "exp084_P_*.json" -ErrorAction SilentlyContinue).Count
$e = @(Get-ChildItem $o -Filter "exp084_E_*.json" -ErrorAction SilentlyContinue).Count
$m = Test-Path (Join-Path $o "exp084_M_s0.json")
$t = @(Get-ChildItem $o -Filter "exp084_train_X_s*.json" -ErrorAction SilentlyContinue).Count
$n = @(Get-Process | Where-Object { $_.ProcessName -match "^python" }).Count
$last = ""
if (Test-Path "D:\gpu083\chain084.log") { $last = (Get-Content "D:\gpu083\chain084.log" -Tail 1) }
"$v|py $n|P $p|E $e|M $m|T $t|$last"
