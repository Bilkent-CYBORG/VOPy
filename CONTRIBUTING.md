# Contributing to VOPy

Thank you for your interest in contributing to VOPy! We cherish contributions from the community.


- [Contributing to VOPy](#contributing-to-vopy)
  - [Getting Started](#getting-started)
    - [Setting Up Your Development Environment](#setting-up-your-development-environment)
    - [Making Changes to the Library](#making-changes-to-the-library)
  - [Adding New Algorithms](#adding-new-algorithms)
      - [1. Create the Algorithm Class](#1-create-the-algorithm-class)
      - [2. Add Algorithm to Package Exports](#2-add-algorithm-to-package-exports)
      - [3. Write Tests](#3-write-tests)
      - [4. Document Your Algorithm](#4-document-your-algorithm)
      - [5. Submit Your Pull Request](#5-submit-your-pull-request)
  - [Code and Docs Styles](#code-and-docs-styles)
  - [Testing](#testing)


## Getting Started

### Setting Up Your Development Environment

1. Fork the repository on GitHub
2. Clone your fork locally:
   ```bash
   git clone https://github.com/YOUR_USERNAME/VOPy.git
   cd VOPy
   ```

3. Set up the development environment using `uv` (recommended) and install pre-commit hooks:
   ```bash
   uv sync --all-extras
   pre-commit install
   ```

### Making Changes to the Library

1. Create a new branch for your changes:
   ```bash
   git checkout -b your-feature-branch
   ```

2. Make your changes and ensure they follow the project's coding standards highlighted in [Code and Docs Styles](#code-and-docs-styles)

3. Run the tests to ensure nothing is broken:
   ```bash
   pytest
   ```

4. Commit your changes with a clear commit message:
   ```bash
   git commit -m "Brief description of your changes"
   ```

5. Push to your fork and submit a pull request

## Adding New Algorithms

If you'd like to contribute a new black-box vector optimization algorithm to VOPy, follow these guidelines:

#### 1. Create the Algorithm Class

- Create your algorithm module in `vopy/algorithms/` (e.g., `your_algorithm.py`)
- Your algorithm class should inherit from `Algorithm` (see [`vopy/algorithms/algorithm.py`](vopy/algorithms/algorithm.py))
- Implement your algorithm with clear docstrings following the format highlighted in [Code and Docs Styles](#code-and-docs-styles)

#### 2. Add Algorithm to Package Exports

Update [init file of algorithms subpackage](vopy/algorithms/__init__.py) to include your new algorithm.

#### 3. Write Tests

- Create a test file in `test/algorithms/` named `test_your_algorithm.py`
- Write comprehensive tests covering all functionality
- Ensure your tests pass with `pytest test/algorithms/test_your_algorithm.py`
- For examples see [tests for other algorithms](test/algorithms/)

#### 4. Document Your Algorithm

- Update the documentation page on [Algorithms](docs/source/algorithms.rst)
- Consider adding an example notebook in `examples/` demonstrating your algorithm, by properly naming your example

#### 5. Submit Your Pull Request

- Ensure all tests pass
- Include a clear description of the algorithm and its use case
- Reference any papers or resources that describe the algorithm
- Be responsive to feedback during code review

## Code and Docs Styles

- Follow PEP 8 style guidelines (use `black` and `usort`, they are also precommit hooks)
- Use type hints
- Write clear, descriptive docstrings following **Sphinx docstring format** for all public classes and methods
  - See existing code in [`vopy/`](vopy/) for examples

## Testing

- Write tests for all new functionality
- Ensure existing tests still pass
- Aim for good test coverage of your code

Run tests with coverage:
```bash
pytest -n auto --cov=vopy -ra
```
<br/>

---

<br/>


<center>
   <b>Thank you for contributing to VOPy!</b>
</center>
