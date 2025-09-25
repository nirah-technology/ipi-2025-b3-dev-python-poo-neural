from abc import ABC, abstractmethod
from typing import List
from math import exp
from random import uniform



class Neurone(ABC):
    def __init__(self, input_neurones_connections_count: int):
        self.weights: List[float] = []

        for _ in range(input_neurones_connections_count):
            weight: float = uniform(0, 1)
            self.weights.append(weight)

        self.bias: float = uniform(0, 1)

    @abstractmethod
    def activate(self, input_value: float) -> float: ...

    @abstractmethod
    def derivate(self, input_value: float) -> float: ...


class ReluNeurone(Neurone):
    def activate(self, input_value: float) -> float:
        return max(0, input_value)

    def derivate(self, input_value: float) -> float:
        return 1 if input_value > 0 else 0


class SigmoidNeurone(Neurone):
    def activate(self, input_value: float) -> float:
        return 1 / (1 + exp(-input_value))

    def derivate(self, input_value: float) -> float:
        return self.activate(input_value) * (1 - self.activate(input_value))


class TanhNeurone(Neurone):
    def activate(self, input_value: float) -> float:
        return (exp(input_value) - exp(-input_value)) / (exp(input_value) + exp(-input_value))

    def derivate(self, input_value: float) -> float:
        return 1 - self.activate(input_value) ** 2



