## Vector Optimization with Active Learning

[![Test Workflow](https://github.com/Bilkent-CYBORG/VOPy/actions/workflows/test.yml/badge.svg)](https://github.com/Bilkent-CYBORG/VOPy/blob/master/.github/workflows/test.yml)
[![Documentation Status](https://readthedocs.org/projects/vopy/badge/?version=latest)](https://vopy.readthedocs.io/en/latest/)
[![License](https://img.shields.io/badge/license-MIT-green.svg)](LICENSE)

### What is VOPy?
VOPy is an open-source Python library built to address noisy black-box vector optimization problems, where the user preferences are encoded with a cone order.

### What to do with VOPy?
VOPy includes several pre-implemented algorithms, models, orders, and problems from the literature for black-box vector optimization, allowing users to select and utilize components based on their specific needs. Specifically, you can:
- Using existing methods for novel problems
- Benchmark novel algorithms with literature
- ... and anything in between utilizing wide range of existing tools!

### How To Start?

Visit our [**website**](https://vopy.readthedocs.io/en/latest/) to see tutorials, examples and API references on how to use VOPy.


### Setup

Installation using pip:
```bash
pip install vopy
```

#### Latest (Unstable) Version
To upgrade to the latest (unstable) version, run

```bash
pip install --upgrade git+https://github.com/Bilkent-CYBORG/VOPy.git
```

#### Manual installation (for development)

If you are contributing a pull request, it is best to perform a manual installation:

```sh
git clone https://github.com/Bilkent-CYBORG/VOPy.git
cd VOPy
mamba env create --name vopy --file environment.yml  # To setup a proper development environment
pip install -e .
```

For all development requirements, see [requirements.txt](requirements.txt) or [environment.yml](environment.yml).
