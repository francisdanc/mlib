package mlib.network.activations;

import mlib.matrixes.InputMatrix;

public interface ActivationFunction {
	public double activate(double a);
	public double[][] activate(double[][] mat);
	public double derivative(double a);
	
}
