# Contributing to `bbrl`

In order to contribute to this repository you will need to use pull requests, see below

To know more about the project go to the [README](README.md) first.

## Pre-commit hooks

Pre-commits hooks have been configured for this project using the 
[pre-commit](https://pre-commit.com/) library:

- [black](https://github.com/psf/black) python formatter
- [flake8](https://flake8.pycqa.org/en/latest/) python linter

To get them going on your side, first install pre-commit:

```bash
pip install pre-commit
```

Then run the following commands from the root directory of this repository:

```bash
pre-commit install
pre-commit run --all-files
```

These pre-commits are applied to all the files, except the directory tmp/
(see .pre-commit-config.yaml)


## Pull Requests

We actively welcome your pull requests.

1. Fork the repo and create your branch from `main`.
2. If you've added code that should be tested, add tests.
3. If you've changed APIs, update the documentation.
4. Ensure the test suite passes.
5. Make sure your code lints.

## Issues

We use GitHub issues to track public bugs. Please ensure your description is
clear and has sufficient instructions to be able to reproduce the issue.

## License
By contributing to `bbrl`, you agree that your contributions will be licensed
under the LICENSE file in the root directory of this source tree.
