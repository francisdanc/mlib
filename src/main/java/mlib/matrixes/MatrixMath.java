package mlib.matrixes;

import java.util.Arrays;

import mlib.network.activations.ActivationFunction;
public class MatrixMath {

		public MatrixMath() {
			
		}
		
		

		public static double[][] dot(double[][] mat1, double[][] mat2) {
			double[][] res = new double[mat1.length][mat2[0].length];
			
			if(mat1[0].length != mat2.length) {
				throw new RuntimeException("The dimensions of the matrixes do not line up\n mat 1 is: " + mat1.length + " " + mat1[0].length + "\nmat2 is: " + mat2.length + " " + mat2[0].length);
			}
			
			
			for(int i = 0; i < mat1.length; i++) {
				for(int j = 0; j < mat2[0].length; j++) {
					double sum = 0.0;
					
					for(int k = 0; k < mat1[0].length; k++) {
						sum += mat1[i][k] * mat2[k][j];
					}
					
					res[i][j] = sum;
				}
			}
			
			
			return res;
		}
		
		public static double[][] hadamard(double[][] one, double[][] two){
			double[][] res = new double[one.length][one[0].length];
			
			if(one.length != two.length || one[0].length != two[0].length) {
				throw new IllegalArgumentException("Input matrices must be of the same dimensions!");
			}
			
			
			for(int i = 0; i < one.length; i++) {
				for (int j = 0; j < one[0].length; j++) {
					res[i][j] = one[i][j] * two[i][j];
				}
			}
			
			
			
			return res;
		}
		
		public static double[][] scalarMultiply(double d, double[][] mat){
			double[][] res = new double[mat.length][mat[0].length];
			
			for(int i = 0; i < mat.length; i++) {
				for(int j = 0; j < mat[0].length; j++) {
					res[i][j] = mat[i][j] * d;
				}
			}
			
			return res;
		}
		
		public static double[][] subtractElementWise(double[][] mat, double[][] mat1) {
			
			
			for(int i = 0; i < mat.length; i++) {
				for(int j = 0; j < mat[0].length; j++) {
					mat[i][j] = mat[i][j] - mat1[i][j];
				}
			}
			
			return mat;
		}
		
		public static double[][] transpose(double[][] matrix) {
		    int rows = matrix.length;
		    int cols = matrix[0].length;
		    double[][] transposed = new double[cols][rows];

		    for (int i = 0; i < rows; i++) {
		        for (int j = 0; j < cols; j++) {
		            transposed[j][i] = matrix[i][j];
		        }
		    }

		    return transposed;
		}
		
		public static double[][] getDerivative(ActivationFunction af, double[][] mat){
			
			for(int i = 0; i < mat.length; i++) {
				for(int j = 0; j < mat[0].length; j++) {
					mat[i][j] = af.derivative(mat[i][j]);
				}
			}
			
			return mat;
		}
		
		
		public static double[][] elementWiseSquare(double[][] mat) {
			
			for(int i = 0; i < mat.length; i++) {
				for(int j = 0; j < mat[0].length; j++) {
					mat[i][j] = mat[i][j] * mat[i][j];
				}
			}
			
			
			return mat;
		}
		
		
		public static double sum(double[][] mat){
			double sum = 0.0;
			
			for(int i = 0; i < mat.length; i++) {
				for(int j = 0; j < mat[0].length; j++) {
					sum += mat[i][j];
				}
			}
			
			
			return sum;
		}
		
		public static double[][]addElements(double[][] mat1, double[]mat2){
			
			double[][] output = new double[mat1.length][mat2.length];
			
			if(mat1[0].length != mat2.length) {
				throw new IllegalArgumentException("addition matrix must equal the size of each row");
			}
			
			for(int i = 0; i < mat1.length; i++) {
				for(int j = 0; j < mat1[0].length; j++) {
					output[i][j] = mat1[i][j] + mat2[j];
				}
			}
			
			return output;
		}
		
		
		public static void main(String args[]) {
			double[][] one = {{1,2,3},{4,5,6}};
			double[][] two = {{7,8},{9,10},{11,12}};
			
			double[][] res = dot(one, two);
			
			for(double[] row : res) {
				for(double d : row) {
					System.out.println(d);
				}
			}
		}
}
