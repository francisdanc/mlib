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
	
	public void train(InputMatrix inputs, double lr, int epochs, InputMatrix expected) {
		if(this.layers.size() < 1) {
			throw new RuntimeException("Model must contain at least one layer");
		}
		
		ActivationFunction sig = new sigmoid();
		
		for(int i = 0; i < layers.size(); i++) {
			layers.get(i).initialiseWeights();
		}
		
		for(int e = 0; e < epochs; e++) {
//		------------------------------------Forward Pass----------------------------------
			Stack<double[][]> activationStack = new Stack<double[][]>();
			for(int i = 0; i < layers.size(); i++) {
				double[][] activations;
				DenseLayer currentLayer = layers.get(i);
				// If the current layer is the input layer then set the inputs to the layer as the inputs to the network
				if(i == 0) {
					currentLayer.setInputs(inputs);
				}else {
					activations = layers.get(i - 1).Sum(sig);
					
					InputMatrix layerInputs = new InputMatrix(activations.length, activations[0].length);
					layerInputs.setInputMatrix(activations);
					currentLayer.setInputs(layerInputs);
					activationStack.push(activations);
				}
			}
	//		----------------------------------------------------------------------------------
			
	//	    -----------------------------------Backwards Pass---------------------------------
			
			LinkedList<double[][]> delta = new LinkedList<double[][]>();
			LinkedList<double[][]> weightGradient = new LinkedList<double[][]>();
			double MSE;
			for(int i = layers.size() - 1; i >=0; i--) {
				
				
				// On the output layer, get the weight gradients directly from the error
				if(i == layers.size() - 1) {
					
					
					double[][] output = layers.getLast().Sum(sig);
					
					if(output.length != expected.getInputMatrix().length || output[0].length != expected.getInputMatrix()[0].length ) {
						throw new RuntimeException("The output matrix and the expected matrix are of different sizes");};
					
					double[][] error = MatrixMath.subtractElementWise(output, expected.getInputMatrix());
					double[][] deltaOut = MatrixMath.hadamard(error, MatrixMath.getDerivative(sig, output));
					delta.offer(deltaOut);
					
					weightGradient.add(MatrixMath.dot(MatrixMath.transpose(activationStack.pop()), deltaOut));
					MSE = MatrixMath.sum(MatrixMath.elementWiseSquare(error)) / (error.length * error[0].length);
					
					if(e % 200 == 0) {
						System.out.printf("Epoch: %d Loss: %.4f %n", e, MSE);
						MatrixMath.printMatrix(output);
					}
				}else if(i < layers.size() - 1 && i > 0) {
					double[][] layerNextWeights = layers.get(i + 1).getWeightMatrix().getWeights();
					double[][] prevDelta = delta.pop();
					double[][] prevActivations = activationStack.pop();
					double[][] hiddenDelta = MatrixMath.hadamard(MatrixMath.dot(prevDelta, MatrixMath.transpose(layerNextWeights)), MatrixMath.getDerivative(sig, prevActivations));
					delta.add(hiddenDelta);
					weightGradient.add(MatrixMath.hadamard(MatrixMath.transpose(prevActivations), hiddenDelta));
					
				}else if(i == 0) {
					double[][] layerNextWeights = layers.get(i + 1).getWeightMatrix().getWeights();
					double[][] prevDelta = delta.pop();
					double[][] inputActivations = layers.get(i).Sum(sig);
					double[][] inputDelta = MatrixMath.hadamard(MatrixMath.dot(prevDelta, MatrixMath.transpose(layerNextWeights)), MatrixMath.getDerivative(sig, inputActivations));
					weightGradient.add(MatrixMath.dot(MatrixMath.transpose(inputs.getInputMatrix()), inputDelta));
				}
				
				
			}
	//		----------------------------------------------------------------------------------
			
//			-------------------------------------------Weight Updates--------------------------
			
			for(int i = 0; i < layers.size(); i++) {
				WeightMatrix currentWeightMatrix = layers.get(i).getWeightMatrix();
//				MatrixMath.printMatrix(currentWeightMatrix.getWeights());
				double[][] update = MatrixMath.scalarMultiply(lr, weightGradient.removeLast());
//				MatrixMath.printMatrix(update);
				
				double[][] newWeights = MatrixMath.subtractElementWise(currentWeightMatrix.getWeights(), update);
				
				currentWeightMatrix.setWeights(newWeights);
			}
			
//			-----------------------------------------------------------------------------------
		}
	}
	}
	
	