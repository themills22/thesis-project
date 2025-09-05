# Command help

I am assuming you are running from root folder of repo.
Replay buffer is set to 100,000 instead of default 1,000,000 as that would cost 4GB of memory, which is a big barrier on my machine.

```powershell
python -m rl_zoo3.train --algo td3 `
    --env EllipseSystemEnv-v0 --env-kwargs max_episode_steps:1000 dimension:10 perturb:0.05 point_count:50000 matrix_count:2500 action_limit:0.05 `
    --hyperparams noise_type:normal noise_std:0.05 buffer_size:100000 policy_kwargs:"dict(net_arch=dict(pi=[500, 500], qf=[500, 500])) `
    --evel-freq 10000 --eval-episodes 10 --n-eval-envs 1 `
    --save-freq 100000
    --conf-file .\python\rl\td3.yml --tensorboard-log .\logs\td3\tensorboard
```
