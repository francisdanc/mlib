package mlib.network.activations;

import mlib.matrixes.InputMatrix;

public interface ActivationFunction {
	public double activate(double a);
	public InputMatrix activate(InputMatrix mat);
	public double derivative(double a);
	
}
