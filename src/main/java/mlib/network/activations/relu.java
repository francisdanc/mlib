package mlib.network.activations;

import mlib.matrixes.InputMatrix;

public class relu implements ActivationFunction {
	
	@Override
	public double activate(double a) {
	    // ReLU activation
	    return (a > 0) ? a : 0.01 * a;
	}

	@Override
	public double[][] activate(double[][] mat) {
	    double[][] activations = new double[mat.length][mat[0].length];

	    for (int i = 0; i < mat.length; i++) {
	        for (int j = 0; j < mat[0].length; j++) {
	            double x = mat[i][j];
	            activations[i][j] = activate(x); // ReLU
	        }
	    }

	    return activations;
	}

	@Override
	public double derivative(double a) {
	    // ReLU derivative — assumes input is pre-activation
	    return (activate(a) > 0) ? 1 : 0.01;
	}

}
