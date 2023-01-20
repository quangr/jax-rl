Some jax Reinforcement Learning implementation.

# ppo_mujoco_envpool_xla_jax

The body is copied from <https://github.com/vwxyzjn/cleanrl/blob/master/cleanrl/ppo_atari_envpool_xla_jax.py> , but this implementation trying to keep the hyperparameter and training process same as SOTA MuJoCo benchmark [tianshou](https://github.com/thu-ml/tianshou).

In this implementation.

- copy runningmeanstd implementation from tianshou, and modified it into jax version

- write env wrapper in jax way.(the api is also different from traditional step and reset, the jax version wrapper should implement reset,send,recv)

- run 8x~20x faster than tianshou

|  Mujoco SPS (global_step per second) | tianshou | jax version | Speedup |
| :-------------: | :---------: | :--------------:|  :--------------:|
|       64 cores Xeon + A100       |   1270    |       19190     | 15.1x |

# Quick Start

## Linux

1. Install required packages

you can following <https://github.com/google/jax#installation> to install jax and jax[cuda](CUDA 11.4 or newer is required,If you have an Ada Lovelace (e.g., RTX 4080) or Hopper (e.g., H100) GPU, you must use CUDA 11.8 or newer.)

Basically, if you has the right cuda version, you only need to run following commands.

`pip install jax`

`pip install --upgrade "jax[cuda]" -f https://storage.googleapis.com/jax-releases/jax_cuda_releases.html`

Then you install envpool and flax

```
pip install envpool==0.8.1
pip install flax
```

you can run `apt-get install libgl1` if you havn't install libgl1 yet.

If you want track your run, you should install wandb.

## Windows

Windows users can use JAX on CPU and GPU via the Windows Subsystem for Linux.

## Using Docker

If you trying to run in vastai, you can try <https://hub.docker.com/r/pyhf/cuda/> mirror.
