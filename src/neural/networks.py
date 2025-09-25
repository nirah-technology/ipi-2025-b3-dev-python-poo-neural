from typing import Optional, List

from .layers import InputLayer, HiddenLayer, OutputLayer
from .neurones import Neurone, SigmoidNeurone

class NeuralNetworkBuilder:
    def __init__(self):
        self.__input_layer: InputLayer|None = None
        self.__hidden_layers: List[HiddenLayer] = []
        self.__output_layer: Optional[OutputLayer] = None

    def with_input_layer(self, neurones_count: int):
        neurones: List[Neurone] = []
        for _ in range(neurones_count):
            neurone = SigmoidNeurone(0)
            neurones.append(neurone)
        self.__input_layer = InputLayer(neurones)
        return self
    
    def with_output_layer(self, neurones_count: int):
        neurones: List[Neurone] = []
        for _ in range(neurones_count):
            neurone = SigmoidNeurone(0)
            neurones.append(neurone)
        self.__output_layer = OutputLayer(neurones)
        return self
    

class NeuralNetwork:
    def __init__(
            self, 
            input_layer: InputLayer, 
            hidden_layers: list[HiddenLayer], 
            output_layer: OutputLayer):

        self.layers = []
        self.layers.append(input_layer)
        self.layers.extend(hidden_layers)
        self.layers.append(output_layer)

    @staticmethod
    def builder():
