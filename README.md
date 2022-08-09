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

* * When using an AutoResetGymAgent and no replay buffer, transitions from an episode to the next were considered as standard steps in an episode. We added a mechanism to properly filter them out, using an additional `get_transitions()` function in the Workspace class.

### Starting with notebooks

- Getting started: The [BBRL model, and a simple agent-environment interaction code](https://colab.research.google.com/drive/1_yp-JKkxh_P8Yhctulqm0IrLbE41oK1p?usp=sharing)

- Building a simple Neural RL agent in interaction with an environment: [A notebook with code](https://colab.research.google.com/drive/1Ui481r47fNHCQsQfKwdoNEVrEiqAEokh?usp=sharing)

- Explanations about [the NoAutoResetGymAgent](https://colab.research.google.com/drive/1EX5O03mmWFp9wCL_Gb_-p08JktfiL2l5?usp=sharing)

- [Dealing with TimeLimits](https://colab.research.google.com/drive/1erLbRKvdkdDy0Zn1X_JhC01s1QAt4BBj?usp=sharing)

- Explanations about [the AutoResetGymAgent, multiple environments and episodes](https://colab.research.google.com/drive/1W9Y-3fa6LsPeR6cBC1vgwBjKfgMwZvP5?usp=sharing)

### Coding your first RL algorithms with BBRL

- [Coding DQN using a NoAutoResetGymAgent](https://colab.research.google.com/drive/1raeuB6uUVUpl-4PLArtiAoGnXj0sGjSV?usp=sharing)

- [Coding DQN using an AutoResetGymAgent](https://colab.research.google.com/drive/1H9_gkenmb_APnbygme1oEdhqMLSDc_bM?usp=sharing)

- [A basic version of the A2C algorithm](https://colab.research.google.com/drive/1yAQlrShysj4Q9EBpYM8pBsp2aXInhP7x?usp=sharing)

- [A more advanced version of the A2C algorithm](https://colab.research.google.com/drive/1C_mgKSTvFEF04qNc_Ljj0cZPucTJDFlO?usp=sharing) dealing with discrete and continuous actions.

### Learning RL with `bbrl` in your favorite coding environment

Have a look at the [bbrl_examples](https://github.com/osigaud/bbrl_example.git) library.

### Code Documentation: (will come soon)

[Read the docs](docs/index.html)

### Development

See [contributing](CONTRIBUTING.md)

## Dependencies

`bbrl` utilizes `pytorch`, `hydra` for configuring experiments, and `gym` for reinforcement learning algorithms.

## License

`bbrl` is released under the MIT license. See [LICENSE](LICENSE) for additional details about it.
