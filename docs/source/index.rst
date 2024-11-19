.. VOPy documentation master file, created by
   sphinx-quickstart on Sun May 12 02:59:36 2024.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

:github_url: https://github.com/Bilkent-CYBORG/VOPy

VOPy's documentation!
=====================================

Welcome to VOPy, an open-source Python library built to tackle the challenges of black-box vector optimization. Designed for scenarios where multiple objectives must be optimized simultaneously, VOPy goes beyond standard multi-objective optimization tools by offering a unique, cone-based ordering of solutions. With features tailored for noisy environments, both discrete and continuous design spaces, limited budgets, and batch observations, VOPy opens up new possibilities for researchers and practitioners. Its modular architecture supports easy integration of existing methods and encourages the creation of innovative algorithms, making VOPy a versatile toolkit for advancing work in vector optimization.


.. toctree::
   :glob:
   :maxdepth: 2
   :caption: Examples

   examples/*

.. toctree::
   :maxdepth: 1
   :caption: Package Reference

   algorithms
   models
   order
   ordering_cone
   acquisition
   design_space
   confidence_region
   datasets
   utils
   

Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`
