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
    
    def with_hidden_layer(self, neurones_count: int):
        neurones: List[Neurone] = []
        neurones_count_from_previous_layer: int = 0

        if len(self.__hidden_layers) > 0:
            neurones_count_from_previous_layer = len(self.__hidden_layers[-1].neurones)
        else:
            neurones_count_from_previous_layer = len(self.__input_layer.neurones)

        for _ in range(neurones_count):
            neurone = SigmoidNeurone(neurones_count_from_previous_layer)
            neurones.append(neurone)


        hidden_layer: HiddenLayer = HiddenLayer(neurones)
        self.__hidden_layers.append(hidden_layer)
        return self

    def with_output_layer(self, neurones_count: int):
        neurones: List[Neurone] = []
        neurones_count_from_previous_layer: int = 0

        if len(self.__hidden_layers) > 0:
            neurones_count_from_previous_layer = len(self.__hidden_layers[-1].neurones)

        else:
            neurones_count_from_previous_layer = len(self.__input_layer.neurones)

        for _ in range(neurones_count):
            neurone = SigmoidNeurone(neurones_count_from_previous_layer)
            neurones.append(neurone)
        self.__output_layer = OutputLayer(neurones)
        return self

    def build(self):
        return NeuralNetwork(self.__input_layer, self.__hidden_layers, self.__output_layer)   

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
        return NeuralNetworkBuilder()
