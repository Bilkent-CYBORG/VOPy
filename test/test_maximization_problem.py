import importlib
from unittest import mock, TestCase

import numpy as np
from vopy.datasets import Dataset
from vopy.maximization_problem import (
    ContinuousProblem,
    DecoupledEvaluationProblem,
    get_continuous_problem,
    ProblemFromDataset,
)

from vopy.utils import set_seed
from vopy.utils.seed import SEED


class TestProblemFromDataset(TestCase):
    """Test the ProblemFromDataset class."""

    def test_evaluate(self):
        set_seed(SEED)

        dataset = mock.Mock(spec=Dataset)

        dataset._in_dim = 2
        dataset._out_dim = 1
        dataset.in_data = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
        dataset.out_data = np.array([[0], [1], [2], [3]])

        problem = ProblemFromDataset(dataset, noise_var=0.1)

        x = np.array([[0.1, 0.1], [0.1, 0.8]])
        y = problem.evaluate(x, noisy=False)
        np.testing.assert_array_equal(y, np.array([[0], [1]]))

        y = problem.evaluate(x, noisy=True)
        self.assertNotEqual(np.prod(y), 0)


class TestDecoupledEvaluationProblem(TestCase):
    """Test the DecoupledEvaluationProblem class."""

    def test_evaluate(self):
        set_seed(SEED)

        x = np.array([[0.1, 0.1], [0.1, 0.8]])
        y = np.array([[0, 1], [2, 3]])

        dataset = mock.Mock(spec=Dataset)

        dataset._in_dim = 2
        dataset._out_dim = 2
        dataset.in_data = x
        dataset.out_data = y

        problem = ProblemFromDataset(dataset, noise_var=0.1)

        decoupled_problem = DecoupledEvaluationProblem(problem)

        # Test kwargs
        y_pred = decoupled_problem.evaluate(x, None, noisy=False)
        np.testing.assert_array_equal(y_pred, y)
        y_pred = decoupled_problem.evaluate(x, None, noisy=True)
        self.assertNotEqual(np.prod(y_pred), 0)

        # Test evaluation index
        y_pred = decoupled_problem.evaluate(x, 0, noisy=False)
        np.testing.assert_array_equal(y_pred, np.array([0, 2]))
        y_pred = decoupled_problem.evaluate(x, [0, 1], noisy=False)
        np.testing.assert_array_equal(y_pred, np.array([0, 3]))


class TestContinuousProblem(TestCase):
    """Test the ContinuousProblem class."""

    def setUp(self):
        set_seed(SEED)

        module = importlib.import_module(name="vopy.maximization_problem")
        module_globals = module.__dict__

        self.problem_names = [
            obj.__name__
            for obj in module_globals.values()
            if isinstance(obj, type)
            and issubclass(obj, ContinuousProblem)
            and obj is not ContinuousProblem
        ]

        self.noise_var = 0.1

    def test_get_continuous_problem(self):
        for name in self.problem_names:
            with self.subTest(name=name):
                problem = get_continuous_problem(name, self.noise_var)
                self.assertIsInstance(problem, ContinuousProblem)

        with self.assertRaises(ValueError):
            get_continuous_problem("weird_problem_name", self.noise_var)

    def test_attributes(self):
        for name in self.problem_names:
            with self.subTest(name=name):
                problem = get_continuous_problem(name, self.noise_var)
                self.assertTrue(hasattr(problem, "out_dim"))

    def test_evaluation(self):
        class MockProblem(ContinuousProblem):
            out_dim = 2

            def __init__(self):
                super().__init__(0.1)

            def evaluate_true(self, x):
                return x

        problem = MockProblem()

        x = np.array([0, 1])
        y_pred = problem.evaluate(x, noisy=False)
        np.testing.assert_array_equal(y_pred, x.reshape(-1, problem.out_dim))

        y_pred = problem.evaluate(x, noisy=True)
        self.assertNotEqual(np.prod(y_pred), 0)
