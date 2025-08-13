package mlib.network.layers;

import mlib.matrixes.InputMatrix;
import mlib.matrixes.MatrixMath;
import mlib.matrixes.WeightMatrix;
import mlib.network.activations.ActivationFunction;
import mlib.network.activations.sigmoid;
public class DenseLayer {
	
	private int height;
	private InputMatrix inputs = null;
	private WeightMatrix weights = null;
	private ActivationFunction af = new sigmoid();
	private int num_inputs;
	
	public DenseLayer(int h, int i) {
		this.height = h;		
		this.num_inputs = i;
	}
	
	
	/**
	 * Input data into the layer.
	 * */
	public void Input(InputMatrix inputs) {
		this.inputs = inputs;
	}
	
	public void initialiseWeights() {

		weights = new WeightMatrix();
		weights.initialiseMatrix(this.num_inputs, this.height);
	}
	/**
	 * Sum the inputs by the weights and apply the layers activation function to them.
	 * */
	public InputMatrix Sum() {
		if(this.inputs == null) {
			throw new RuntimeException("The inputs for this layer have not been initialised.");
		}
		double[][] raw = MatrixMath.dot(inputs.getInputMatrix(), weights.getWeights());
		
		InputMatrix output = new InputMatrix(raw.length, raw[0].length);
		
		output.setInputMatrix(raw);
		
		return output;
	}


	public int getHeight() {
		return height;
	}


	public void setHeight(int height) {
		this.height = height;
	}


	public InputMatrix getInputs() {
		return inputs;
	}


	public void setInputs(InputMatrix inputs) {
		this.inputs = inputs;
	}


	public ActivationFunction getAf() {
		return af;
	}


	public void setAf(ActivationFunction af) {
		this.af = af;
	}


	public WeightMatrix getWeightMatrix() {
		return weights;
	}


	public void setWeights(WeightMatrix weights) {
		this.weights = weights;
	}
	
	
	
	
}
