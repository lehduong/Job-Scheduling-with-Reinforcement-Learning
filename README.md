<p align="center">
<img src="assets/logo.png" style="max-width:100px"/>
<br>
<a href="https://github.com/lehduong/Job-Scheduling-with-Reinforcement-Learning/blob/master/LICENSE">
    <img alt="GitHub" src="https://img.shields.io/github/license/lehduong/Job-Scheduling-with-Reinforcement-Learning">
</a>
<img alt="python" src="https://img.shields.io/badge/python-%3E%3D3.6-blue">
<img alt="tensorflow" src="https://img.shields.io/badge/tensorflow-%3E%3D2.2.0-orange">
<img alt="ubuntu" src="https://img.shields.io/badge/ubuntu-%3E%3D18.04-yellowgreen">
</p>
</h1>

<p align="center">
Learning to Assign Credit in Input-driven Environment (LACIE) reduce the variance of estimation of advantages value in noisy MDP with hindsight distribution.
</p>

## Input-driven MDP
Input-driven MDP are the Markov processes governed by not only agent's actions but also stochastic, exogenous input processes [1]. These environments have high variance inheritantly making it hard to learn optimal policy.

This repository implemented:

+ Input-dependence baseline as in proposed in [1].

+ **Lacie** - an algorithm that learn to weight the advantages of each rollout in hindsight with respect to future input sequences.

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

4. Install [Park Platform](https://github.com/park-project/park). I modified the platform slightly to make it compatible with OpenAI's baseline.
```bash
git clone https://github.com/lehduong/park &&\
cd park && \
pip install -e .
```

## Run experiments
See `scripts` for examples.

## Reference

 [1] [Variance Reduction for Reinforcement Learning in Input-Driven Environments](https://openreview.net/forum?id=Hyg1G2AqtQ).

## Acknowledgement
The started code is based on [ikostrikov's repository](https://github.com/ikostrikov/pytorch-a2c-ppo-acktr-gail).

