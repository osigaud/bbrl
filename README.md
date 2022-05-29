# BBRL

`bbrl`- A Flexible and Simple Library for Reinforcement Learning inspired from SaLinA

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

* May 2022:
* * First commit of the BBRL repository

* March 2022:
* * Forked SaLinA and started to modify the model

## Detailed documentation with notebooks

- Getting started: The [BBRL model, and a simple agent-environment interaction](https://colab.research.google.com/drive/1_yp-JKkxh_P8Yhctulqm0IrLbE41oK1p?usp=sharing)

- Building a simple Neural RL agent in interaction with an environment: A [notebook with code](https://colab.research.google.com/drive/1Ui481r47fNHCQsQfKwdoNEVrEiqAEokh?usp=sharing)

- More details about the interaction model: [AutoResetGymAgent, multiple environments and episodes](https://colab.research.google.com/drive/1W9Y-3fa6LsPeR6cBC1vgwBjKfgMwZvP5?usp=sharing)


## Learning RL with `bbrl` in your favorite coding environment

Have a look at the [bbrl_examples](https://github.com/osigaud/bbrl_example.git) library.

## Code Documentation: (will come soon)

[Read the docs](https://bbrl.readthedocs.io/en/latest/)

**For development, see [contributing](CONTRIBUTING.md)**

## Dependencies

`bbrl` utilizes `pytorch`, `hydra` for configuring experiments, and `gym` for reinforcement learning algorithms.

## Note on the logger

We provide a simple Logger that logs in both tensorboard format, but also as pickle files that can be re-read to make tables and figures. See [logger](bbrl/utils/logger.py). This logger can be easily replaced by any other logger.

## License

`bbrl` is released under the MIT license. See [LICENSE](LICENSE) for additional details about it.
