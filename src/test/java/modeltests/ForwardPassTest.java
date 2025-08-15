package modeltests;

import static org.junit.jupiter.api.Assertions.assertEquals;

import java.util.Stack;

import org.junit.jupiter.api.Test;

import mlib.matrixes.InputMatrix;
import mlib.matrixes.MatrixMath;
import mlib.matrixes.WeightMatrix;
import mlib.model.Model;
import mlib.network.activations.*;
import mlib.network.layers.DenseLayer;
public class ForwardPassTest {
//	TODO: Fix the logic of this test. It needs to use a custom set of weights so we know exactly what the output should be 
//	and test directly against that.
	
	
//	Model model = new Model();
//	InputMatrix input;
//	InputMatrix expected;
//	DenseLayer h1;
//	DenseLayer out;
	public ForwardPassTest(){
//		this.h1 = new DenseLayer(2,2);
//		this.out = new DenseLayer(1,2);
//		
//		
//		
//		this.model.addLayer(h1);
//		this.model.addLayer(out);
//		
//		this.input = new InputMatrix(4, 2);
//		double[][] inputs = {{0,0}, {0,1}, {1,0}, {1,1}};
//		input.setInputMatrix(inputs);
//		this.expected = new InputMatrix(4,1);
//		double[][] e = {{1}, {0}, {0}, {1}};
//		expected.setInputMatrix(e);
//		h1.setInputs(this.input);
//		this.h1.initialiseWeights();
//		this.out.initialiseWeights();
	}
	
	@Test
	public void testBackwardsPass() {
		
	}
	
	@Test
	public void testForwardPass() {
		
		Model model = new Model();
		InputMatrix input = new InputMatrix(4,2);
		DenseLayer h1 = new DenseLayer(2, 2);
		DenseLayer o = new DenseLayer(1, 2);
		model.addLayer(h1);
		model.addLayer(o);
	    // Define input matrix: 4 samples, 2 features
	    double[][] inputs = {
	        {0, 0},
	        {0, 1},
	        {1, 0},
	        {1, 1}
	    };
	    input.setInputMatrix(inputs);

	    // Manually set weights and biases for h1 (2 inputs → 2 neurons)
	    double[][] h1Weights = {
	        {1.0, 0.5},  // Neuron 1 weights
	        {-1.0, 0.5}    // Neuron 2 weights
	    };
	    double[] h1Biases = {0.0, 0.0};

	    WeightMatrix weights = new WeightMatrix();
	    weights.setWeights(h1Weights);
	    h1.setWeights(weights);
	    h1.setBias(h1Biases);

	    // Skip output layer setup — not used in forward pass

	    // Run forward pass
	    Stack<double[][]> activations = model.ForwardPass(input);
	    double[][] hiddenActivations = activations.pop();  // Only hidden layer output

	    // Manually compute expected hidden layer activations
	    sigmoid sig = new sigmoid();
	    double[][] expectedHidden = new double[4][2];
	    for (int i = 0; i < inputs.length; i++) {
	        double h1n1 = inputs[i][0] * h1Weights[0][0] + inputs[i][1] * h1Weights[1][0] + h1Biases[0];
	        double h1n2 = inputs[i][0] * h1Weights[0][1] + inputs[i][1] * h1Weights[1][1] + h1Biases[1];
	        expectedHidden[i][0] = sig.activate(h1n1);
	        expectedHidden[i][1] = sig.activate(h1n2);
	    }

	    double[][] dot = MatrixMath.addElements(MatrixMath.dot(inputs, h1Weights), h1Biases);
	    double[][] dota = new double[dot.length][dot[0].length];
	    MatrixMath.printMatrix(dot);
	    for(int i = 0; i < dot.length; i++) {
	    	for(int j = 0; j < dot[0].length; j++) {
	    		dota[i][j] = sig.activate(dot[i][j]);
	    	}
	    }

	    // Compare model output to expected hidden activations
	    for (int i = 0; i < hiddenActivations.length; i++) {
	        for (int j = 0; j < hiddenActivations[0].length; j++) {
	            assertEquals(expectedHidden[i][j], hiddenActivations[i][j], 1e-6,
	                "Mismatch at [" + i + "][" + j + "]");
	        }
	    }
	}
}
