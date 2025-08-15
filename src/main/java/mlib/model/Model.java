package mlib.model;

import mlib.network.activations.sigmoid;
import mlib.network.layers.DenseLayer;

import java.util.Arrays;
import java.util.LinkedList;
import java.util.ArrayList;
import java.util.Random;
import java.util.Stack;

import mlib.matrixes.WeightMatrix;
import mlib.matrixes.InputMatrix;
import mlib.matrixes.MatrixMath;
import mlib.network.activations.*;
public class Model {
	private ArrayList<DenseLayer> layers = new ArrayList<DenseLayer>();
	
	
	public Model() {}


	public ArrayList<DenseLayer> getLayers() {
		return layers;
	}


	public void setLayers(ArrayList<DenseLayer> layers) {
		this.layers = layers;
	};
	
	public void addLayer(DenseLayer layer) {
		this.layers.add(layer);
	}
	
	
	public Stack<double[][]> ForwardPass(InputMatrix inputs, int e, ActivationFunction af){
	if(e % 500 == 0) {	

	}
			
			Stack<double[][]> weightedSumStack = new Stack<double[][]>();
			double[][] inputWeightedSums = new double[inputs.getInputMatrix().length][inputs.getInputMatrix()[0].length];
			for(int i = 0; i < layers.size(); i++) {
				double[][] weightedSums;
				DenseLayer currentLayer = layers.get(i);
				// If the current layer is the input layer then set the inputs to the layer as the inputs to the network
				if(i == 0) {
					currentLayer.setInputs(inputs);
					weightedSums = currentLayer.Sum();
					weightedSumStack.push(weightedSums);
					
				}else {
					double[][] prevWeightedSums = layers.get(i - 1).Sum();
					
					InputMatrix layerInputs = new InputMatrix(prevWeightedSums.length, prevWeightedSums[0].length);
					layerInputs.setInputMatrix(af.activate(prevWeightedSums));
					currentLayer.setInputs(layerInputs);
					weightedSums = currentLayer.Sum();
					if(i != layers.size() - 1) {
						weightedSumStack.push(weightedSums);
					}
				}
			}

			return weightedSumStack;
		} 
	
	

	private double[][] BackwardsPass(Stack<double[][]> weightedSumStack, InputMatrix expected, InputMatrix inputs, int e, double lr, ActivationFunction af) {
		
		int DebugWhen = 100;
		
	
		
		LinkedList<double[][]> delta = new LinkedList<double[][]>(); // Store delta calculations in LinkedList
		LinkedList<double[][]> weightGradient = new LinkedList<double[][]>(); // Store weightGradient calculations in LinkedList
		double MSE = 0.0;
		double[][] output = new double[expected.getBatchSize()][expected.getFeatureLength()]; // Output is same size as expected
		
		
		for(int i = layers.size() - 1; i >=0; i--) {
			
			
			// On the output layer, get the weight gradients directly from the error
			if(i == layers.size() - 1) {
				

				double[][] weightedSumsOutput = layers.getLast().Sum(); // Activations of the output layer, which is the prediction
				output = af.activate(weightedSumsOutput);

				
				if(output.length != expected.getInputMatrix().length || output[0].length != expected.getInputMatrix()[0].length ) {
					throw new RuntimeException("The output matrix and the expected matrix are of different sizes");
				}
				
				// Calculate the error by subtracting the expected from the output
				double[][] error = MatrixMath.subtractElementWise(output, expected.getInputMatrix());
				

				double[][] derivative = MatrixMath.getDerivative(af, weightedSumsOutput);
				// Get the delta of the output by multiplying the derivative of the output by the error
				double[][] deltaOut = MatrixMath.hadamard(error, derivative);

				
				// Add the delta to the end of the delta LinkedList
				delta.add(deltaOut);
				
				double[] currentBias = layers.get(i).getBias();
				double[] biasUpdate = new double[currentBias.length];

				// Accumulate gradients for each bias neuron
				for (int k = 0; k < currentBias.length; k++) {
				    double sumDelta = 0.0;
				    for (int j = 0; j < deltaOut.length; j++) {
				        sumDelta += deltaOut[j][k];
				    }
				    // Average over batch and apply learning rate
				    biasUpdate[k] = currentBias[k] - (lr * (sumDelta / deltaOut.length));
				}

				layers.getLast().setBias(biasUpdate);

				double[][] previousWeightedSums = weightedSumStack.pop(); // Weighted sums of the previous layer (in the xor case the weighted sums of the hidden layer)
				
				// Calculate the weight gradient as the transpose of the previous layers activations multiplied with the deltaOutput
				weightGradient.add(MatrixMath.dot(MatrixMath.transpose(af.activate(previousWeightedSums)), deltaOut));
				
				
				MSE = MatrixMath.sum(MatrixMath.elementWiseSquare(error)) / error.length; 
				

			}else if(i < layers.size() - 1 && i > 0) {
				
		
				double[][] layerNextWeights = layers.get(i + 1).getWeightMatrix().getWeights();
	
				double[][] prevDelta = delta.pop();
				
				double[][] prevWeightedSums = weightedSumStack.pop();

		
				double[][] prevActivations = af.activate(prevWeightedSums);
				double[][] derivative = MatrixMath.getDerivative(af, layers.get(i).Sum());

	
				double[][] hiddenDelta = MatrixMath.hadamard(MatrixMath.dot(prevDelta, MatrixMath.transpose(layerNextWeights)), derivative);
				
				delta.add(hiddenDelta);
				
				weightGradient.add(MatrixMath.dot(MatrixMath.transpose(prevActivations), hiddenDelta));
				
				double[] currentBias = layers.get(i).getBias();
				double[] biasUpdate = new double[currentBias.length];
				
				for (int b = 0; b < currentBias.length; b++) {
				    // sum over all rows (samples) in inputDelta
				    double sumDelta = 0;
				    for (int sample = 0; sample < hiddenDelta.length; sample++) {
				        sumDelta += hiddenDelta[sample][b];
				    }
				    // gradient step: subtract learning rate * gradient
				    biasUpdate[b] = currentBias[b] - lr * (sumDelta / hiddenDelta.length);
				}

				layers.get(i).setBias(biasUpdate);
			}else if(i == 0) {
				
				// Get weights of the next layer
				double[][] layerNextWeights = layers.get(i + 1).getWeightMatrix().getWeights();
				// Get the previously calculated delta

				double[][] nextDelta = delta.pop();
				// Get the activations of the input layer
				double[][] inputWeightedSums = layers.get(i).Sum();
				double[][] derivative =  MatrixMath.getDerivative(af, inputWeightedSums);
				// Calculate the delta of this layer
				double[][] inputDelta = MatrixMath.hadamard(MatrixMath.dot(nextDelta, MatrixMath.transpose(layerNextWeights)), derivative);
				// Calculate 
				double[] currentBias = layers.get(i).getBias();
				double[] biasUpdate = new double[currentBias.length];
				
				for (int b = 0; b < currentBias.length; b++) {
				    // sum over all rows (samples) in inputDelta
				    double sumDelta = 0;
				    for (int sample = 0; sample < inputDelta.length; sample++) {
				        sumDelta += inputDelta[sample][b];
				    }
				    // gradient step: subtract learning rate * gradient
				    biasUpdate[b] = currentBias[b] - lr * (sumDelta / inputDelta.length);
				}

				layers.get(i).setBias(biasUpdate);
				
				weightGradient.add(MatrixMath.dot(MatrixMath.transpose(inputs.getInputMatrix()), inputDelta));

			}
			
			
		}

		// Update weights
		for(int i = 0; i < layers.size(); i++) {
			WeightMatrix currentWeightMatrix = layers.get(i).getWeightMatrix();

			double[][] update = MatrixMath.scalarMultiply(lr, weightGradient.removeLast());

			
			double[][] newWeights = MatrixMath.subtractElementWise(currentWeightMatrix.getWeights(), update);
			
			currentWeightMatrix.setWeights(newWeights);
			layers.get(i).setWeights(currentWeightMatrix);
		}
		if(e % 5000 == 0) {
			System.out.println("Epoch: " + e);
			System.out.printf("Loss: %f%n" , MSE);
		}
		
		return output;
		
	}
	
	public void train(InputMatrix inputs, double lr, int epochs, InputMatrix expected, ActivationFunction af) {
		if(this.layers.size() < 1) {
			throw new RuntimeException("Model must contain at least one layer");
		}
		
		
		
		for(int i = 0; i < layers.size(); i++) {
			layers.get(i).initialiseWeights();
		}
		double[][] output = new double[expected.getInputMatrix().length][expected.getInputMatrix().length];
		for(int e = 0; e < epochs; e++) {

//		------------------------------------Forward Pass----------------------------------
			Stack<double[][]> activationStack = ForwardPass(inputs, e, af);
	//		----------------------------------------------------------------------------------

	//	    -----------------------------------Backwards Pass---------------------------------
			
//			TODO: Figure out how to fix the MSE dependency
			output = BackwardsPass(activationStack, expected, inputs, e, lr, af);

	//		----------------------------------------------------------------------------------
			

		}
//		double[][] output = layers.getLast().Sum(sig);
		MatrixMath.printMatrix(output);
	}
	}
	
	