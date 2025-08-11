package mlib.matrixes;

public class InputMatrix extends Matrix{
	private double[][] matrix;
	private final int bs;

	private final int fl;
	
	
	public InputMatrix(int batchsize, int featurelength) {	
		this.matrix = new double[batchsize][featurelength];	
		this.bs = batchsize;
		this.fl = featurelength;
	}
	
	public void setInputMatrix(double[][] m) {
		this.matrix = m;
	}
	
	public double[][] getInputMatrix(){
		return this.matrix;
	}
	
	public int getBatchSize() {
		return bs;
	}

	public int getFeatureLength() {
		return fl;
	}

}
