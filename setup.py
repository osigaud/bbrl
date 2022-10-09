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
    version="0.1.4",
    install_requires=open("requirements.txt", "r").read().splitlines(),
    tests_require=["pytest==4.4.1"],
    test_suite="tests",
    description="RL library inspired from salina",
    author="Olivier Sigaud",
    author_email="Olivier.Sigaud@isir.upmc.fr",
    url="https://github.com/osigaud/bbrl.git",
    license="MIT",
)
