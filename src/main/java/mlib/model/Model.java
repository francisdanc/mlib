package mlib.model;

import mlib.network.activations.sigmoid;
import mlib.network.layers.DenseLayer;

import java.util.Arrays;
import java.util.LinkedList;
import java.util.ArrayList;
import java.util.Random;
import java.util.Stack;

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
		// Set the activation function to sigmoid  and intialise the connection between the first hidden layer and the inputs
		ActivationFunction sig = new sigmoid();
		DenseLayer inputlayer = layers.getFirst();
		LinkedList<InputMatrix> weighted_sums = new LinkedList<InputMatrix>();
		LinkedList<InputMatrix> activations = new LinkedList<InputMatrix>(); // The stack of activations needed to calculate derivatives for backpropogation
		inputlayer.setInputs(inputs);
		inputlayer.initialiseWeights();

		//System.out.println(inputlayer.getInputs().getBatchSize() + " " + inputlayer.getInputs().getFeatureLength());
		
		// Initialise all of the weights for each layer in the network
		for(DenseLayer layer : layers) {
			layer.initialiseWeights();
		}
		
		for(int e = 0; e < epochs; e++) {
			// Clean up the stacks at the end of each loop to make sure that each iteration has a fresh stack
			weighted_sums.clear();
			activations.clear();
			
//--------------------------------------------------------Forward Pass--------------------------------------------------------------	
			
			/*Get all of the activations from every layer*/
			InputMatrix ws_current;
			InputMatrix a_current;
			ws_current = inputlayer.Sum();
			a_current = sig.activate(ws_current);
			weighted_sums.push(ws_current); // raw weighted sums from the input layer get pushed to the bottom of the stack
			activations.push(a_current); // activations from the input layer get pushed to the bottom of the stack
			
			
			
			for(int i = 1; i < layers.size() ; i++) {
				//Start at the output layer
				

				
				DenseLayer current_layer = layers.get(i);
					
				current_layer.setInputs(a_current);
	
				ws_current = current_layer.Sum();
				a_current = sig.activate(ws_current);
					
				weighted_sums.offer(ws_current); // output layer activations and weighted sums go to the top of the queue
				activations.offer(a_current);
				
			}
			
			/*
			 * Queue:
			 * Back
			 * -------
			 * Output activations 
			 * Input activations
			 * -------
			 * Front
			 * */
//----------------------------------------------------------------------------------------------------------------------------------
			
			
			//System.out.println(inputlayer.getInputs().getBatchSize() + " " + inputlayer.getInputs().getFeatureLength());
			
			// Get the activations and weighted sums going from output to input by popping them from the stack
			InputMatrix prevActivations = activations.poll(); // Input activations 
			
			DenseLayer outputLayer = layers.getLast();
			
		    if(outputLayer.getInputs().getFeatureLength() != expected.getFeatureLength()) {
		        throw new RuntimeException("Output layer and expected output sizes do not match");
		    };
		    
		    
		    // --------------------------------- Calculate output delta ------------------------------
		    InputMatrix error = new InputMatrix(outputLayer.getInputs().getBatchSize(), outputLayer.getInputs().getFeatureLength());
		    error.setInputMatrix(MatrixMath.subtractElementWise(MatrixMath.cloneMatrix(outputLayer.getInputs().getInputMatrix()), expected.getInputMatrix()));
		    
		    double MSE = (MatrixMath.sum(MatrixMath.elementWiseSquare(error.getInputMatrix()))) / (prevActivations.getBatchSize() * prevActivations.getFeatureLength());
		    
		    InputMatrix outputDerivative = new InputMatrix(prevActivations.getBatchSize(), prevActivations.getFeatureLength());
		    outputDerivative.setInputMatrix(MatrixMath.getDerivative(sig, prevActivations.getInputMatrix())); 
		    
		    InputMatrix delta = new InputMatrix(prevActivations.getBatchSize(), prevActivations.getFeatureLength());
		    delta.setInputMatrix(MatrixMath.hadamard(error.getInputMatrix(), outputDerivative.getInputMatrix()));
		    // ----------------------------------------------------------------------------------------
		    
		    
		    
		    Stack<double[][]> weightedGradients = new Stack<double[][]>();
		    
		 // ----------------------------Calculate the weight gradient for the output layer-----------------------
		    System.out.println("The activations of the output layer");
		    System.out.println("");
		    MatrixMath.printMatrix(prevActivations.getInputMatrix());
	    	double[][] prev_activations_T = MatrixMath.transpose(prevActivations.getInputMatrix());
	    	System.out.println("");
	    	MatrixMath.printMatrix(prev_activations_T);
	    	double[][] w_grad =  MatrixMath.scalarMultiply(lr, MatrixMath.dot(prev_activations_T, delta.getInputMatrix()));
	    	weightedGradients.push(w_grad); // Push the output layers gradients to the top of the stack
	    	System.out.println("The w_gradients for the output layer vs the output layers weight matrix");
	    	MatrixMath.printMatrix(w_grad);
	    	System.out.println("");
	    	MatrixMath.printMatrix(layers.getLast().getWeightMatrix().getWeights());
	    // -----------------------------------------------------------------------------------------------------
		  
//		    if(e % 200 == 0) {
//		    	System.out.printf("epoch: %d loss: %f%n", e, MSE);
//		    	System.out.println("activations: ");
//		    	for(int i = 0; i < prevActivations.getInputMatrix().length; i++) {
//		    		System.out.printf("%s %n", Arrays.toString(prevActivations.getInputMatrix()[i]));
//		    	}
//		    }
	    	
	    	// Go from the input layer to the last hidden layer
		    for(int i = 0 ; i < layers.size() - 2; i++) {
		    	DenseLayer layer = layers.get(i); 
		    	
		    	
		    	
		    	if(i ==0 ) {
		    		prevActivations = sig.activate(weighted_sums.pop()); // If the first hidden layer get the activations of the weighted sums from the inputs
		    	}else {
		    		prevActivations = activations.poll(); 
		    	}
		    	
		    	
		    	
		    	
//----------------------------------------------Calculate new delta---------------------------------------------------------------
		    	double[][] next_weights = layers.get(i + 1).getWeightMatrix().getWeights();
		    	double[][] weights_T = MatrixMath.transpose(next_weights);
		    		
		    		//MatrixMath.printMatrix(prevWs.getInputMatrix());
		    	double[][] currentDerivative = MatrixMath.getDerivative(sig, prevActivations.getInputMatrix());
//		    		MatrixMath.printMatrix(currentDerivative);
//		    		MatrixMath.printMatrix(MatrixMath.dot(delta.getInputMatrix(), weights_T));
		    	double[][] newDelta = MatrixMath.hadamard(MatrixMath.dot(delta.getInputMatrix(), weights_T), currentDerivative);
		    	delta.setInputMatrix(newDelta);
//--------------------------------------------------------------------------------------------------------------------------------		    		

		    	
		    	
//--------------------------------------------Calculate weighted gradients for hidden layer using new delta-----------------------		    	
		    	prev_activations_T = MatrixMath.transpose(prevActivations.getInputMatrix());
		    	System.out.println("Transposed activations of the current hidden layer");
		    	MatrixMath.printMatrix(prev_activations_T);
		    	w_grad =  MatrixMath.scalarMultiply(lr, MatrixMath.dot(prev_activations_T, delta.getInputMatrix()));
		    	System.out.println("The weighted gradients vs the hidden layer");
		    	MatrixMath.printMatrix(w_grad);
		    	System.out.println("%n");
		    	MatrixMath.printMatrix(layers.get(i).getWeightMatrix().getWeights());
		    	
		    	weightedGradients.push(w_grad);

//--------------------------------------------------------------------------------------------------------------------------------
		    	
		    	
		    	double[] biasGradients = new double[layers.get(i).getHeight()];
		    	for(int k = 0; k < layers.get(i).getHeight(); k++) {
		    		int batchSize = layers.get(i).getInputs().getBatchSize();
		    		double sum = 0.0;
		    		for(int b = 0; b < batchSize; b++) {
		    			sum += delta.getInputMatrix()[b][k];
		    		}
		    		biasGradients[k] = sum / batchSize;	    		
		    	}
		    	
		    	double[] biases = layer.getBias();
		    	for(int b = 0; b < biases.length; b++) {
		    	    biases[b] -= lr * biasGradients[b];
		    	}
		    	layer.setBias(biases);
		    	
		    	
		    	
		    	
		    }
		    
		    // Update the weights
		    
		    for(int i = 0; i < layers.size(); i++) {
		    	DenseLayer layer = layers.get(i);
		    	double[][] currentWeights = layer.getWeightMatrix().getWeights();
//		    	for(int s = 0; s < weightedGradients.size(); s++) {
//		    		System.out.println("Stack element: " + e);
//		    		MatrixMath.printMatrix(weightedGradients.get(e));
//		    	}
		    	double[][] grad = weightedGradients.pop();
		    	//double[][] update = MatrixMath.scalarMultiply(lr, grad);
		    	System.out.println("Current weights: ");
		    	MatrixMath.printMatrix(currentWeights);
		    	System.out.println("w_grad: ");
		    	MatrixMath.printMatrix(w_grad);
		    	layer.getWeightMatrix().setWeights(MatrixMath.subtractElementWise(currentWeights, grad));
		    	
		    	
		    	//System.out.println(Arrays.toString(layers.get(i).getBias()));
		    }
		    
		}
	    
		}
	}
	
	