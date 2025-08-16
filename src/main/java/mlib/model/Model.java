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

		LinkedList<double[][]> delta = new LinkedList<double[][]>(); 
		LinkedList<double[][]> weightGradient = new LinkedList<double[][]>(); 
		double MSE = 0.0;
		double[][] output = new double[expected.getBatchSize()][expected.getFeatureLength()]; 
		
		
		for(int i = layers.size() - 1; i >=0; i--) {
			
			

			if(i == layers.size() - 1) {
				

				double[][] weightedSumsOutput = layers.getLast().Sum(); 
				output = af.activate(weightedSumsOutput);

				
				if(output.length != expected.getInputMatrix().length || output[0].length != expected.getInputMatrix()[0].length ) {
					throw new RuntimeException("The output matrix and the expected matrix are of different sizes");
				}
				

				double[][] error = MatrixMath.subtractElementWise(output, expected.getInputMatrix());
				

				double[][] derivative = MatrixMath.getDerivative(af, weightedSumsOutput);

				double[][] deltaOut = MatrixMath.hadamard(error, derivative);

				delta.add(deltaOut);
				
				double[] currentBias = layers.get(i).getBias();
				double[] biasUpdate = new double[currentBias.length];

				for (int k = 0; k < currentBias.length; k++) {
				    double sumDelta = 0.0;
				    for (int j = 0; j < deltaOut.length; j++) {
				        sumDelta += deltaOut[j][k];
				    }

				    biasUpdate[k] = currentBias[k] - (lr * (sumDelta / deltaOut.length));
				}

				layers.getLast().setBias(biasUpdate);

				double[][] previousWeightedSums = weightedSumStack.pop(); // Weighted sums of the previous layer (in the xor case the weighted sums of the hidden layer)
				
		
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

				    double sumDelta = 0;
				    for (int sample = 0; sample < hiddenDelta.length; sample++) {
				        sumDelta += hiddenDelta[sample][b];
				    }

				    biasUpdate[b] = currentBias[b] - lr * (sumDelta / hiddenDelta.length);
				}

				layers.get(i).setBias(biasUpdate);
			}else if(i == 0) {
				
		
				double[][] layerNextWeights = layers.get(i + 1).getWeightMatrix().getWeights();

				double[][] nextDelta = delta.pop();

				double[][] inputWeightedSums = layers.get(i).Sum();
				double[][] derivative =  MatrixMath.getDerivative(af, inputWeightedSums);

				double[][] inputDelta = MatrixMath.hadamard(MatrixMath.dot(nextDelta, MatrixMath.transpose(layerNextWeights)), derivative);

				double[] currentBias = layers.get(i).getBias();
				double[] biasUpdate = new double[currentBias.length];
				
				for (int b = 0; b < currentBias.length; b++) {
				   
				    double sumDelta = 0;
				    for (int sample = 0; sample < inputDelta.length; sample++) {
				        sumDelta += inputDelta[sample][b];
				    }
				
				    biasUpdate[b] = currentBias[b] - lr * (sumDelta / inputDelta.length);
				}

				layers.get(i).setBias(biasUpdate);
				
				weightGradient.add(MatrixMath.dot(MatrixMath.transpose(inputs.getInputMatrix()), inputDelta));

			}
			
			
		}

		
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


		Stack<double[][]> activationStack = ForwardPass(inputs, e, af);

		output = BackwardsPass(activationStack, expected, inputs, e, lr, af);

		
	}
		MatrixMath.printMatrix(output);
	}
	}
	
	