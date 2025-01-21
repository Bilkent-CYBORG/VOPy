## VOPy: A Framework for Black-box Vector Optimization

[![Test Workflow](https://github.com/Bilkent-CYBORG/VOPy/actions/workflows/test.yml/badge.svg)](https://github.com/Bilkent-CYBORG/VOPy/blob/master/.github/workflows/test.yml)
[![Coverage Status](https://coveralls.io/repos/github/Bilkent-CYBORG/VOPy/badge.svg)](https://coveralls.io/github/Bilkent-CYBORG/VOPy)
[![Documentation Status](https://readthedocs.org/projects/vopy/badge/?version=latest)](https://vopy.readthedocs.io/en/latest/)
[![License](https://img.shields.io/badge/license-MIT-green.svg)](LICENSE)

### What is VOPy?
VOPy is an open-source Python library built to address noisy black-box [vector optimization](https://en.wikipedia.org/wiki/Vector_optimization) problems, where the user preferences are encoded with a cone order.

<p align="center">
  <img src="docs/source/_static/vopy_deps.jpg" alt="Overview of the dependencies, core modules, and built-in algorithms of VOPy" width="600px">
</p>

### What to do with VOPy?
VOPy includes several pre-implemented algorithms, models, orders, and problems from the literature for black-box vector optimization, allowing users to select and utilize components based on their specific needs. Specifically, you can:
- Use existing black-box vector optimization methods for new problems
- Benchmark novel algorithms with the state-of-the-art
- Utilize a wide range of existing tools in black-box vector optimization

### How To Start?

Visit our [**website**](https://vopy.readthedocs.io/en/latest/) to see tutorials, examples and API references on how to use VOPy. We recommend starting out with the [**motivating example**](https://vopy.readthedocs.io/en/latest/examples/01_motivating_example.html).


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

Further, installing the pre-commit hooks are **highly** encouraged.

```sh
# Inside the package folder
pre-commit install
```

### **Citing**

If you use VOPy, please cite the following paper:

```
@article{yildirim2024vopy,
  title={{VOPy}: A Framework for Black-box Vector Optimization},
  author={Yıldırım, Yaşar Cahit and Karagözlü, Efe Mert and Korkmaz, İlter Onat and Ararat, Çağın and Tekin, Cem},
  journal={arXiv preprint arXiv:2412.06604},
  year={2024}
}
```

### **License**

VOPy is under [MIT license](LICENSE).
