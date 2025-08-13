package mlib.network.activations;

import mlib.matrixes.InputMatrix;

public class sigmoid implements ActivationFunction {

	@Override
	public double activate(double a) {
		
		return 1.0 / (1.0 + Math.exp(-a));

	}
	
	@Override
	public InputMatrix activate(InputMatrix mat){
		
		for(int i = 0; i < mat.getInputMatrix().length; i++) {
			for(int j = 0; j < mat.getInputMatrix()[0].length; j++) {
				mat.getInputMatrix()[i][j] = activate(mat.getInputMatrix()[i][j]);
			}
		}
		
		
		return mat;
	}
	
	@Override
	public double derivative(double a) {
		return activate(a) * (1 - activate(a));
	}

}
