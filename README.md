# BBRL

`bbrl`- A Flexible and Simple Library for Reinforcement Learning deriving from SaLinA

BBRL stands for "BlackBoard Reinforcement Learning". Initially, this library was a fork of [the SaLinA library](https://github.com/facebookresearch/salina). 
But SaLinA is a general model for sequential learning whereas BBRL is dedicated to RL, thus it focuses on a subset of SaLinA. 
Morevover, BBRL is designed for education purpose (in particular, to teach various RL algorithms, concepts and phenomena). 
Thus the fork slowly drifted away from SaLinA and became independent after a few months, even if some parts of the code are still inherited from SaLinA.

## TL;DR.

`bbrl` is a lightweight library extending PyTorch modules for developping **Reinforcement Learning** models
* It supports simultaneously training with AutoReset on multiple environments
* It works on multiple CPUs and GPUs

## Citing `bbrl`

BBRL being inspired from SaLinA, please use this bibtex if you want to cite BBRL in your publications:

Link to the paper: [SaLinA: Sequential Learning of Agents](https://arxiv.org/abs/2110.07910)

```
    @misc{salina,
        author = {Ludovic Denoyer, Alfredo de la Fuente, Song Duong, Jean-Baptiste Gaya, Pierre-Alexandre Kamienny, Daniel H. Thompson},
        title = {SaLinA: Sequential Learning of Agents},
        year = {2021},
        publisher = {Arxiv},
        howpublished = {\url{https://github.com/facebookresearch/salina}},
    }

```

## Quick Start

* Just clone the repo
* `pip install -e .`


## News

* May-June 2023:
* * Integrated the use of gymnasium. Turned google colab notebooks into colab compatible jupyter notebooks. Refactored all the notebooks. 

* August 2022:
* * Major updates of the notebook-based documentation

* May 2022:
* * First commit of the BBRL repository

* March 2022:
* * Forked SaLinA and started to modify the model

## Documentation

### Main differences to SaLinA

- BBRL only contains core classes to implement RL algorithms.

- Because both notations coexist in the literature, the GymAgent classes support the case where doing action $a_t$ in state $s_t$ results in reward $r_t$, and the case where it results in reward $r_{t+1}$.

- Some output string were corrected, some variable were renamed and some comments were improved to favor code readability.

- A few small bugs in SaLinA were fixed:

* * The replay buffer was rejecting samples that did not fit inside when the number of added samples was beyond the limit. This has been corrected to implement the standard FIFO behavior of replay buffer.

* * When using autoreset=True and no replay buffer, transitions from an episode to the next were considered as standard steps in an episode. We added a mechanism to properly filter them out, using an additional `get_transitions()` function in the Workspace class.

### Starting with notebooks

There are two kinds of notebooks:

- some are just for reading, they explain the main concepts and design choices behind BBRL

- some are designed for learning about RL algorithms and practicing with BBRL. Ideally, these notebooks should be studied in the provided order.

#### Explanatory notebooks

- Getting started: The [BBRL model](https://colab.research.google.com/drive/1_yp-JKkxh_P8Yhctulqm0IrLbE41oK1p?usp=sharing)

- Explanations about using [autoreset=False](https://colab.research.google.com/drive/1EX5O03mmWFp9wCL_Gb_-p08JktfiL2l5?usp=sharing)

- Explanations about using [autoreset=True](https://colab.research.google.com/drive/1W9Y-3fa6LsPeR6cBC1vgwBjKfgMwZvP5?usp=sharing)

- [Different Reward Indexing Schemes](https://colab.research.google.com/drive/1Cld72_FBA1aMS2U4EsyV3LGZIlQC_PsC?usp=sharing)

- [Dealing with Time Limits](https://colab.research.google.com/drive/1erLbRKvdkdDy0Zn1X_JhC01s1QAt4BBj?usp=sharing)

#### Coding your first RL algorithms with BBRL

Most of the notebooks below can be run under jupyter notebook as well as under Google colaboratory. In any case, download it on your disk and run it with your favorite notebook environment.

- [A simple agent-environment interaction code](https://colab.research.google.com/drive/1gSdkOBPkIQi_my9TtwJ-qWZQS0b2X7jt?usp=sharing)

- [Building a simple Neural RL agent in interaction with a gym environment](https://colab.research.google.com/drive/1Ui481r47fNHCQsQfKwdoNEVrEiqAEokh?usp=sharing)

- [Coding DQN using autoreset=False](http://master-dac.isir.upmc.fr/rld/rl/03-1-dqn-introduction.student.ipynb)

- [Coding DQN using autoreset=True](http://master-dac.isir.upmc.fr/rld/rl/03-2-dqn-full.student.ipynb)

- [Coding DPPG and TD3 using autoreset=True](http://master-dac.isir.upmc.fr/rld/rl/04-ddpg-td3.student.ipynb)

- [Coding basic Policy Gradient algorithms and REINFORCE](http://master-dac.isir.upmc.fr/rld/rl/05-reinforce.student.ipynb)

- [A basic version of the A2C algorithm](http://master-dac.isir.upmc.fr/rld/rl/06-1-a2c_basic.student.ipynb)

- [A more advanced version of the A2C algorithm](http://master-dac.isir.upmc.fr/rld/rl/06-2-a2c_advanced.student.ipynb)

- [The KL penalty version of PPO](http://master-dac.isir.upmc.fr/rld/rl/07-1-ppo_penalty.student.ipynb)

- [The clipped version of PPO](http://master-dac.isir.upmc.fr/rld/rl/07-2-ppo_clip.student.ipynb)

- [Coding SAC](http://master-dac.isir.upmc.fr/rld/rl/08-sac.student.ipynb)

- [Coding TQC](https://colab.research.google.com/drive/1Lg9_M9YwI_E6Xm1on8GY9TLYxLItTSuw?usp=sharing)

### Learning RL with `bbrl` in your favorite coding environment

Have a look at the [bbrl_examples](https://github.com/osigaud/bbrl_example.git) library.

### Code Documentation:

[Generated with pdoc](https://htmlpreview.github.io/?https://github.com/osigaud/bbrl/blob/master/docs/bbrl/index.html)

### Development

See [contributing](CONTRIBUTING.md)

## Dependencies

`bbrl` utilizes `pytorch`, `hydra` for configuring experiments, and `gym` or `gymnasium` for reinforcement learning algorithms.

## License

`bbrl` is released under the MIT license. See [LICENSE](LICENSE) for additional details about it.
