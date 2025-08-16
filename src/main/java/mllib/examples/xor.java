package mllib.examples;

import mlib.matrixes.InputMatrix;
import mlib.model.Model;
import mlib.network.layers.DenseLayer;
import mlib.network.activations.*;
public class xor {
/*		 
 * 		TODO: Create functionality to import datasets
 * 		
 * */
		public static void main(String args[]) {
			Model model = new Model();
			DenseLayer h1 = new DenseLayer(3,2);
			DenseLayer h2 = new DenseLayer(2, 3);
			DenseLayer h3 = new DenseLayer(2, 2);
			DenseLayer out = new DenseLayer(1,2);
			ActivationFunction af = new sigmoid();
			
			
			model.addLayer(h1);
			model.addLayer(h2);
			model.addLayer(h3);
			model.addLayer(out);
			
			InputMatrix input = new InputMatrix(4, 2);
			double[][] inputs = {{0.0,0.0}, {0.0,1.0}, {1.0,0.0}, {1.0,1.0}};
			input.setInputMatrix(inputs);
			InputMatrix expected = new InputMatrix(4,1);
			double[][] e = {{0}, {1}, {1}, {0.0}};
			expected.setInputMatrix(e);
		
			
			model.train(input, 1, 10000, expected, af);
		}
		

}
