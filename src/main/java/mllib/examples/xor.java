package mllib.examples;

import mlib.matrixes.InputMatrix;
import mlib.model.Model;
import mlib.network.layers.DenseLayer;

public class xor {

		public static void main(String args[]) {
			Model model = new Model();
			DenseLayer h1 = new DenseLayer(2,2);
			DenseLayer out = new DenseLayer(1,2);
			
			
			
			model.addLayer(h1);
			model.addLayer(out);
			
			InputMatrix input = new InputMatrix(4, 2);
			double[][] inputs = {{0,0}, {0,1}, {1,0}, {1,1}};
			input.setInputMatrix(inputs);
			InputMatrix expected = new InputMatrix(4,1);
			double[][] e = {{1}, {0}, {0}, {1}};
			expected.setInputMatrix(e);
		
			
			model.train(input, 0.01, 50000, expected);
		}
		

}
