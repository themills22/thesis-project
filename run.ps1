$dataFolder = Get-Item D:\deep-reinforcement-learning\thesis-project\data\power-flow\graph\4
$guessesFolder = Get-Item D:\deep-reinforcement-learning\thesis-project\data\power-flow\graph\4\guesses
$judgedFolder = Get-Item D:\deep-reinforcement-learning\thesis-project\data\power-flow\graph\4\judged
$modelFolder = Get-Item D:\deep-reinforcement-learning\thesis-project\model\power-flow\graph\4
$optimizeFile = Get-Item .\python\optimizers\power_flow_optimizer.py
$trainFile = Get-Item .\python\nn\net_train.py
$judgeFile = Get-Item .\python\optimizers\judge.py
$iteration = 1
while ($true)
{
    $modelFile = Get-ChildItem -File $modelFolder | Sort-Object LastWriteTime | Select-Object -Last 1
    python $optimizeFile.FullName `
        --size 4 `
        --results-folder $guessesFolder.FullName `
        --model-to-load $modelFile.Fullname `
        --count-cutoff 18 `
        --improved-system-cutoff 10000 `
        --input-norm-cap 300

    julia --project=D:\deep-reinforcement-learning\thesis-project\julia .\julia\power_flow_judge.jl

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
        --model-to-load $modelFile.FullName `
        --epochs 2 `
        --epoch-save 2 `
        --print-interval 50000 `
        --input-norm-cap 300

        Write-Host "Finished iteration $iteration"
        $iteration++
}