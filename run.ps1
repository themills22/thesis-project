$dataFolder = Get-Item D:\deep-reinforcement-learning\thesis-project\data\power-flow\graph\4
$guessesFolder = Get-Item D:\deep-reinforcement-learning\thesis-project\data\power-flow\graph\4\guesses
$judgedFolder = Get-Item D:\deep-reinforcement-learning\thesis-project\data\power-flow\graph\4\judged
$modelFolder = Get-Item D:\deep-reinforcement-learning\thesis-project\model\power-flow\graph\4
$optimizeFile = Get-Item .\python\optimizers\power_flow_optimizer.py
$trainFile = Get-Item .\python\nn\net_train.py
$judgeFile = Get-Item .\python\optimizers\judge.py

$numThreads = 8
$iteration = 1
while ($iteration -le 5)
{
    $modelFile = Get-ChildItem -File $modelFolder | Sort-Object LastWriteTime | Select-Object -Last 1
    python $optimizeFile.FullName `
        --size 4 `
        --results-folder $guessesFolder.FullName `
        --model-to-load $modelFile.Fullname `
        --count-cutoff 18 `
        --systems-per-file 500000 `
        --total-systems 500000 `
        --cpu-count $numThreads
        # --input-norm-squared-cap 100000000

    julia --project=D:\deep-reinforcement-learning\thesis-project\julia --threads=$numThreads .\julia\power_flow_judge.jl

    $judgedFile = Get-ChildItem -File $judgedFolder | Sort-Object LastWriteTime | Select-Object -Last 1
    python $judgeFile `
        --count-cutoff 18 `
        $judgedFile.FullName

    python $trainFile.FullName `
        --batch-size 64 `
        --type graph `
        --size 4 `
        --data-folder $dataFolder.FullName `
        --data-folder $judgedFolder.FullName `
        --model-folder $modelFolder.FullName `
        --epochs 1 `
        --epoch-save 1 `
        --print-interval 50000 `
        --input-norm-cap 10000000000

    Write-Host "Finished iteration $iteration"
    $iteration++
}