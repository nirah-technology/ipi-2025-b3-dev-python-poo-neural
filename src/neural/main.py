from .networks import NeuralNetwork
from random import randint

def create_random_neural_network() -> NeuralNetwork:
    min_neurones_count_for_input_layer: int = 2
    max_neurones_count_for_input_layer: int = 10
    min_neurones_count_for_output_layer: int = 2
    max_neurones_count_for_output_layer: int = 10
    min_neurones_count_for_hidden_layer: int = 2
    max_neurones_count_for_hidden_layer: int = 10

    builder = NeuralNetwork.builder()
    input_neurones_count: int = randint(
        min_neurones_count_for_input_layer, 
        max_neurones_count_for_input_layer)
    builder.with_input_layer(input_neurones_count)
    for _ in range(randint(0, 10)):
        hidden_neurones_count: int = randint(
            min_neurones_count_for_hidden_layer, 
            max_neurones_count_for_hidden_layer)
        builder.with_hidden_layer(hidden_neurones_count)
    output_neurones_count: int = randint(
        min_neurones_count_for_output_layer, 
        max_neurones_count_for_output_layer)
    builder.with_output_layer(output_neurones_count)
    return builder.build()


def create_neural_network_generator():
    while True:
        print("Création du 1er réseau de neuronnes...")
        neural_network: NeuralNetwork = create_random_neural_network()
        # del neural_network
        yield neural_network
        print("Création du 2eme réseau de neuronnes")
        neural_network = create_random_neural_network()
        yield neural_network
        print("Création du 3eme réseau de neuronnes")
        neural_network = create_random_neural_network()

def main():
    # genetaror = create_neural_network_generator()

    # for _ in range(100):
    #     print(next(genetaror))
    

if __name__ == "__main__":
    main()
