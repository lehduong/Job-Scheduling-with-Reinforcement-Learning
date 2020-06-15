# Input-Dependent-Baseline

This is the **unofficial** Pytorch implementation of Input-dependence baseline as in [Variance Reduction for Reinforcement Learning in Input-Driven Environments](https://openreview.net/forum?id=Hyg1G2AqtQ).

## Install Dependencies

1. Install Pytorch 

```bash
pip install torch torchvision
```

2. install Tensorflow 2

```bash
pip install tensorflow=2.2
```
or 
```bash
pip install tensorflow-gpu=2.2
```

3. Install [OpenAI baseline](https://github.com/openai/baselines/tree/tf2) (Tensorflow 2 version)
```bash
git clone https://github.com/openai/baselines.git -b tf2 && \
cd baselines && \
pip install -e .
```

**Note**: I haven't tested the code on Tensorflow 1 yet but it should work as well.

4. Install gym
```bash
pip install 'gym[atari]'
```

5. Install [Park Platform](https://github.com/park-project/park). I modified the platform slightly to make it compatible with OpenAI's baseline.
```bash
git clone https://github.com/lehduong/park &&\
cd park && \
pip install -e .
```

## Run experiments
```bash 
python main.py --num-process 16 --recurrent-policy --num-inner-steps 5 --no-cuda --eval-interval 3 --use-linear-lr-decayrameskip-v4
```

# Acknowledgement

