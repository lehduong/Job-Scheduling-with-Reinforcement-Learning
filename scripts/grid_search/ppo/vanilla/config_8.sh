python main.py --num-stream-jobs 1000 --num-stream-jobs-factor 1.05\
                --num-curriculum-time 1 \
                --algo ppo \
                --clip-param 0.2\
                --ppo-epoch 8\
                --num-env-steps 50000000\
                --gamma 1\
                --entropy-coef 0.01\
                --regularize-coef 1\
                --load-balance-service-rates 0.15 0.25 0.35 0.45 0.55 0.65 0.75 0.85 0.95 1.05 \
                --reward-norm-factor 10000\
                --lr 0.000125\
                --num-mini-batch 4\
                --num-process 16 --num-steps 1000 --log-interval 5 \
                --seed 100 --use-memory-to-pred-weights --use-linear-lr-decay\
                --log-dir ppo_8                   