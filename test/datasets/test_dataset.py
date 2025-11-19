import importlib
import unittest

import numpy as np
from vopy.datasets import Dataset, get_dataset_instance


class TestDataset(unittest.TestCase):
    """
    Test cases for the Dataset derivative classes and the get_dataset_instance function.
    """

    def setUp(self):
        module = importlib.import_module(name="vopy.datasets.dataset")
        module_globals = module.__dict__

        self.dataset_names = [
            obj.__name__
            for obj in module_globals.values()
            if isinstance(obj, type) and issubclass(obj, Dataset) and obj is not Dataset
        ]

    def test_get_dataset_instance(self):
        for name in self.dataset_names:
            with self.subTest(name=name):
                dataset = get_dataset_instance(name)
                self.assertIsInstance(dataset, Dataset)

        with self.assertRaises(ValueError):
            get_dataset_instance("weird_dataset_name")

    def test_dataset_attributes(self):
        for name in self.dataset_names:
            with self.subTest(name=name):
                dataset = get_dataset_instance(name)
                self.assertTrue(hasattr(dataset, "_in_dim"))
                self.assertTrue(hasattr(dataset, "_out_dim"))
                self.assertTrue(hasattr(dataset, "_cardinality"))

                np.testing.assert_allclose(
                    (np.min(dataset.in_data), np.max(dataset.in_data)), (0, 1), atol=1e-6
                )
                np.testing.assert_allclose(
                    (np.mean(dataset.out_data), np.var(dataset.out_data)), (0, 1), atol=1e-6
                )
