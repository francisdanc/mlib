package mlib.network.activations;

import mlib.matrixes.InputMatrix;

public class sigmoid implements ActivationFunction {

	@Override
	public double activate(double a) {
		
		return 1.0 / (1.0 + Math.exp(-a));

	}
	
	@Override
	public InputMatrix activate(InputMatrix mat){
		
		InputMatrix activations = new InputMatrix(mat.getBatchSize(), mat.getFeatureLength());
		
		for(int i = 0; i < mat.getInputMatrix().length; i++) {
			for(int j = 0; j < mat.getInputMatrix()[0].length; j++) {
				
				activations.getInputMatrix()[i][j] = activate(mat.getInputMatrix()[i][j]);
			}
		}
		
		
		return activations;
	}
	
	@Override
	public double derivative(double a) {
		
		return a * (1 - a);
	}

}
