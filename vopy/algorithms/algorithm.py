from abc import ABC, abstractmethod


class Algorithm(ABC):
    """
    Abstract base class for all algorithms.
    """

    def __init__(self) -> None:
        super().__init__()

    @abstractmethod
    def run_one_step(self) -> bool:
        pass


class PALAlgorithm(Algorithm, ABC):
    """
    Base class for algorithms that does (\\epsilon, \\delta)-PAC modelling.

    :param epsilon: Determines the accuracy of the PAC-learning framework.
    :type epsilon: float
    :param delta: Determines the success probability of the PAC-learning framework.
    :type delta: float
    """

    def __init__(self, epsilon: float, delta: float) -> None:
        super().__init__()

        self.epsilon = epsilon
        self.delta = delta

        # TODO: Maybe make sample_count an ABC property. Will every child have it?

    @abstractmethod
    def run_one_step(self) -> bool:
        pass
