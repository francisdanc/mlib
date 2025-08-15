package mlib.network.activations;

import mlib.matrixes.InputMatrix;

public class sigmoid implements ActivationFunction {

	@Override
	public double activate(double a) {
		
		return 1.0 / (1.0 + Math.exp(-a));

	}
	
	@Override
	public double[][] activate(double[][] mat){
		
		double[][] activations = new double[mat.length][mat[0].length];
		
		for(int i = 0; i < mat.length; i++) {
			for(int j = 0; j < mat[0].length; j++) {
				
				double x = activate(mat[i][j]);
	            
				
				activations[i][j] = x;
			}
		}
		
		
		return activations;
	}
	
	@Override
	public double derivative(double a) {
		
		return activate(a) * (1 - activate(a));
	}

}
