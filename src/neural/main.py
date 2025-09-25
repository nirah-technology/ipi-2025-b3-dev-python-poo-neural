from .networks import NeuralNetwork

def main():
    neural_network = NeuralNetwork.builder() \
        .with_input_layer(10) \
        .with_hidden_layer(10) \
        .with_hidden_layer(10) \
        .with_hidden_layer(10) \
        .with_hidden_layer(10) \
        .with_hidden_layer(10) \
        .with_output_layer(2) \
        .build()
    
    for layer in neural_network.layers:
        for neurone in layer.neurones:
            print(neurone.weights)


if __name__ == "__main__":
    main()
