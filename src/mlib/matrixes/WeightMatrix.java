package mlib.matrixes;

import mlib.network.layers.DenseLayer;
import java.util.Random;

public class WeightMatrix extends Matrix{
	private double[][] weights;
	
	public WeightMatrix() {}
	
	/**
	 * Initialises the weight matrix of dimensions NxH where N is the number of input neurons and H is the height of the 
	 * input layer.
	 * 
	 * */
	public void initialiseMatrix(int inputs, int height) {	
		double[][] w = new double[inputs][height];
		Random r = new Random();
		for(double[] row: w) {
			for(int i = 0; i < row.length; i++) {
				row[i] = r.nextDouble();
			}
		}
		this.setWeights(w);
	}

	public double[][] getWeights() {
		return weights;
	}

	public void setWeights(double[][] weights) {
		this.weights = weights;
	}
	
	
}
