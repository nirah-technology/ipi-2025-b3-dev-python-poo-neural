from abc import ABC, abstractmethod
from typing import List, Protocol

from .neurones import Neurone

class Layer(ABC):
    def __init__(self, neurones: List[Neurone]):
        self.neurones: List[Neurone] = neurones
    
class InputLayer(Layer):
    def __init__(self, neurones: List[Neurone]):
        super().__init__(neurones)

class HiddenLayer(Layer):
    def __init__(self, neurones: List[Neurone]):
        super().__init__(neurones)

class OutputLayer(Layer):
    def __init__(self, neurones: List[Neurone]):
        super().__init__(neurones)
