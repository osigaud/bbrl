#
# Copyright (c) Sorbonne Universite
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
#

from setuptools import find_packages, setup

setup(
    name="bbrl",
    packages=[package for package in find_packages() if package.startswith("bbrl")],
    version="0.0.1",
    install_requires=[
        "torch>=1.9.0"
        "torchvision"
        "gym==0.21.0"
        "tensorboard"
        "tqdm"
        "hydra-core"
        "numpy"
        "pandas"
        "opencv-python"
        "xformers>=0.0.3"
        "omegaconf"
        "matplotlib"
    ],
    tests_require=["pytest==4.4.1"],
    test_suite="tests",
    description="RL library inspired from salina",
    author="Olivier Sigaud",
    license="MIT",
)
