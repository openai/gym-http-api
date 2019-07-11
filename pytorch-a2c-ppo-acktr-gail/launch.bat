REM Specify number of processes at command line, or default to 1
if [%1]==[] (SET /A num_proc=1) else (SET /A num_proc=%1)
REM Specify environment at command line, or default to "SonicTheHedgehog-Genesis"
if [%2]==[] (SET env="SonicTheHedgehog-Genesis") else (SET env="%2")

REM python main.py --env-name %env% --algo ppo --use-gae --lr 2.5e-4 --clip-param 0.1 --value-loss-coef 0.5 --num-processes %num_proc% --num-steps 128 --num-mini-batch 4 --log-interval 1 --use-linear-lr-decay --entropy-coef 0.01 --save-dir "save" --save-interval 5

REM Settings from https://github.com/aborghi/retro_contest_agent/blob/master/metalearner/ppo2ttifrutti_agent.py
REM num-steps halved from 2048 to 1024 because of limited CUDA video memory
python main.py --env-name %env% --algo ppo --use-gae --lr 2e-4 --clip-param 0.1 --value-loss-coef 0.5 --num-processes %num_proc% --num-steps 1024 --num-mini-batch 16 --log-interval 1 --use-linear-lr-decay --entropy-coef 0.01 --save-dir "save" --save-interval 5 --gamma 0.99 --gae-lambda 0.95 --ppo-epoch 4 --eval-interval 5 --log-dir "/tmp/gym/"