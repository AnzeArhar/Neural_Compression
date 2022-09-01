package network

import (
	"math"
	"math/rand"
)

type Network struct {
	Output           [][]float64
	Weights          [][][]float64
	Bias             [][]float64
	ErrorSignal      [][]float64
	OutputDerivative [][]float64

	NETWORK_LAYER_SIZES []int
	INPUT_SIZE          int
	OUTPUT_SIZE         int
	NETWORK_SIZE        int
}

func (network *Network) Init(layerSizes []int) {
	network.NETWORK_LAYER_SIZES = layerSizes
	network.INPUT_SIZE = network.NETWORK_LAYER_SIZES[0]
	network.NETWORK_SIZE = len(layerSizes)
	network.OUTPUT_SIZE = network.NETWORK_LAYER_SIZES[network.NETWORK_SIZE-1]

	network.Output = make([][]float64, network.NETWORK_SIZE)
	network.Weights = make([][][]float64, network.NETWORK_SIZE)
	network.Bias = make([][]float64, network.NETWORK_SIZE)
	network.ErrorSignal = make([][]float64, network.NETWORK_SIZE)
	network.OutputDerivative = make([][]float64, network.NETWORK_SIZE)

	for i := 0; i < network.NETWORK_SIZE; i++ {
		network.Output[i] = make([]float64, network.NETWORK_LAYER_SIZES[i])
		network.ErrorSignal[i] = make([]float64, network.NETWORK_LAYER_SIZES[i])
		network.OutputDerivative[i] = make([]float64, network.NETWORK_LAYER_SIZES[i])
		network.Bias[i] = randomArray(network.NETWORK_LAYER_SIZES[i])
		if i > 0 {
			network.Weights[i] = make([][]float64, network.NETWORK_LAYER_SIZES[i])
			for j := 0; j < network.NETWORK_LAYER_SIZES[i]; j++ {
				network.Weights[i][j] = randomArray(network.NETWORK_LAYER_SIZES[i-1])
			}
		}
	}
}

func (network *Network) Calculate(input []float64) []float64 {
	network.Output[0] = input
	for layer := 1; layer < network.NETWORK_SIZE; layer++ {
		for neuron := 0; neuron < network.NETWORK_LAYER_SIZES[layer]; neuron++ {
			sum := network.Bias[layer][neuron]
			for previousNeuron := 0; previousNeuron < network.NETWORK_LAYER_SIZES[layer-1]; previousNeuron++ {
				sum += network.Output[layer-1][previousNeuron] * network.Weights[layer][neuron][previousNeuron]
			}
			network.Output[layer][neuron] = sigmoid(sum)
			network.OutputDerivative[layer][neuron] = network.Output[layer][neuron] * (1 - network.Output[layer][neuron])
		}
	}
	return network.Output[network.NETWORK_SIZE-1]
}

func (network *Network) Train(input []float64, target []float64, eta float64) {
	network.Calculate(input)
	network.backpropagationError(target)
	network.updateWeights(eta)
}

func (network *Network) backpropagationError(target []float64) {
	for neuron := 0; neuron < network.NETWORK_LAYER_SIZES[network.NETWORK_SIZE-1]; neuron++ {
		network.ErrorSignal[network.NETWORK_SIZE-1][neuron] =
			(network.Output[network.NETWORK_SIZE-1][neuron] - target[neuron]) *
				network.OutputDerivative[network.NETWORK_SIZE-1][neuron]
	}
	for layer := network.NETWORK_SIZE - 2; layer > 0; layer-- {
		for neuron := 0; neuron < network.NETWORK_LAYER_SIZES[layer]; neuron++ {
			var sum float64
			for nextNeuron := 0; nextNeuron < network.NETWORK_LAYER_SIZES[layer+1]; nextNeuron++ {
				sum += network.Weights[layer+1][nextNeuron][neuron] * network.ErrorSignal[layer+1][nextNeuron]
			}
			network.ErrorSignal[layer][neuron] = sum * network.OutputDerivative[layer][neuron]
		}
	}
}

func (network *Network) updateWeights(eta float64) {
	for layer := 1; layer < network.NETWORK_SIZE; layer++ {
		for neuron := 0; neuron < network.NETWORK_LAYER_SIZES[layer]; neuron++ {
			for previousNeuron := 0; previousNeuron < network.NETWORK_LAYER_SIZES[layer-1]; previousNeuron++ {
				delta := -eta * network.Output[layer-1][previousNeuron] * network.ErrorSignal[layer][neuron]
				network.Weights[layer][neuron][previousNeuron] += delta
			}
			network.Bias[layer][neuron] += -eta * network.ErrorSignal[layer][neuron]
		}
	}
}

func sigmoid(x float64) float64 {
	return 1 / (1 + math.Exp(-x))
}

func randomArray(length int) []float64 {
	array := make([]float64, length)
	for i := 0; i < length; i++ {
		array[i] = rand.Float64()*2 - 1
	}
	return array
}
