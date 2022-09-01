package main

import (
	"encoding/csv"
	"fmt"
	"math"
	"neuralnetwork/network"
	"os"
	"strconv"
	"strings"
)

func main() {
	fmt.Println("Loading mnist dataset...")

	file, _ := os.Open("./mnist/mnist_train.csv")
	reader := csv.NewReader(file)
	trainData, _ := reader.ReadAll()
	file, _ = os.Open("./mnist/mnist_test.csv")
	reader = csv.NewReader(file)
	testData, _ := reader.ReadAll()

	trainInputs := make([][]float64, len(trainData))
	trainTargets := make([][]float64, len(trainData))
	testInputs := make([][]float64, len(testData))
	testTargets := make([][]float64, len(testData))

	for i := 0; i < len(trainData); i++ {
		trainInputs[i] = make([]float64, 28*28)
		trainTargets[i] = make([]float64, 10)
		temp, _ := strconv.ParseInt(trainData[i][0], 10, 16)
		trainTargets[i][temp] = 1
		for j := 0; j < 28; j++ {
			for k := 0; k < 28; k++ {
				temp, _ = strconv.ParseInt(trainData[i][j*28+k], 10, 16)
				trainInputs[i][j*28+k] = float64(temp) / 255
			}
		}
	}
	for i := 0; i < len(testData); i++ {
		testInputs[i] = make([]float64, 28*28)
		testTargets[i] = make([]float64, 10)
		temp, _ := strconv.ParseInt(testData[i][0], 10, 16)
		testTargets[i][temp] = 1
		for j := 0; j < 28; j++ {
			for k := 0; k < 28; k++ {
				temp, _ = strconv.ParseInt(testData[i][j*28+k], 10, 16)
				testInputs[i][j*28+k] = float64(temp) / 255
			}
		}
	}

	fmt.Println("Finished loading dataset")

	//layers := []int{28 * 28, 500, 350, 500, 28 * 28}
	layers := []int{28 * 28, 350, 80, 350, 28 * 28}
	network := network.Network{}
	network.Init(layers)

	fmt.Println("Training the network...")

	learningRate := 0.1 //0.3
	repetitions := 1

	for i := 0; i < len(trainData)*repetitions; i++ {
		//network.Train(trainInputs[i % len(trainData)], trainTargets[i % len(trainData)], learningRate)
		network.Train(trainInputs[i%len(trainData)], trainInputs[i%len(trainData)], learningRate)
		if i%(len(trainData)*repetitions/100) == 0 {
			fmt.Printf("\tProgress: %.2f\n", float64(i)/float64(len(trainData)*repetitions))
		}
	}

	fmt.Println("Finished training the network")

	/*
		fmt.Println("Gathering results...")

		correct := 0
		for i := 0; i < len(testData); i++ {
			output := network.Calculate(testInputs[i])
			max := 0.0
			maxIdx := 0
			for j := 0; j < 10; j++ {
				if output[j] > max {
					max = output[j]
					maxIdx = j
				}
			}
			result := make([]int, 10)
			result[maxIdx] = 1

			compare := 0
			for j := 0; j < 10; j++ {
				if float64(result[j]) != testTargets[i][j] {
					break
				}
				compare++
			}
			if compare == 10 {
				correct++
			}
		}

		fmt.Println("Results:")
		fmt.Printf("\tClassification accuracy: %.3f\n", float64(correct)/float64(len(testData)))
		fmt.Printf("\tClassification error: %.3f\n", 1-float64(correct)/float64(len(testData)))
		fmt.Printf("\t%d out of %d correctly identified\n", correct, len(testData))
	*/

	outFile, _ := os.Create("./out.txt")
	for i := 0; i < 16; i++ {
		in := testInputs[i]
		out := network.Calculate(in)

		for j := 0; j < len(in); j++ {
			in[j] = math.Round(in[j] * 255)
			out[j] = math.Round(out[j] * 255)
		}

		outFile.WriteString(strings.Trim(strings.Join(strings.Fields(fmt.Sprint(in)), ", "), "[]"))
		outFile.WriteString("\n")
		outFile.WriteString(strings.Trim(strings.Join(strings.Fields(fmt.Sprint(out)), ", "), "[]"))
		outFile.WriteString("\n")
	}
}
