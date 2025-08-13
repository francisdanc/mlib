package modeltests;

import static org.junit.jupiter.api.Assertions.*;

import java.util.Arrays;

import org.junit.jupiter.api.Test;

import mlib.matrixes.InputMatrix;
import mlib.matrixes.WeightMatrix;
import mlib.network.activations.sigmoid;
import mlib.network.layers.DenseLayer;
import mlib.matrixes.MatrixMath;

public class OneEpochTest {

    @Test
    public void testOneEpochTrainingStep() {
        // Input: 1 sample, 2 features (e.g., XOR: [1, 0])
        double[][] inputData = {{1.0, 1.0}};
        double[][] expectedOutput = {{1.0}}; // target

        // Initial weights: 2 inputs → 1 neuron
        double[][] weightData = {{0.5}, {-0.5}};
        double[] biasData = {0.0};

        // Create layer
        DenseLayer layer = new DenseLayer(1, 2); // 1 neuron, 2 inputs
        layer.setBias(biasData);
        WeightMatrix wm = new WeightMatrix();
        wm.setWeights(weightData);
        layer.setWeights(wm);

        // Set input
        InputMatrix inputMatrix = new InputMatrix(1, 2);
        inputMatrix.setInputMatrix(inputData);
        layer.Input(inputMatrix);

        // Forward pass
        InputMatrix rawOutput = layer.Sum();
        InputMatrix activatedOutput = layer.getAf().activate(rawOutput);
        System.out.println("Activated output before computing error");
        MatrixMath.printMatrix(activatedOutput.getInputMatrix());
        // Compute error
        double[][] activations_clone = MatrixMath.cloneMatrix(activatedOutput.getInputMatrix());
        double[][] error = MatrixMath.subtractElementWise(
            activations_clone,
            expectedOutput
        );
        System.out.println("Activated output after computing error");
        MatrixMath.printMatrix(activatedOutput.getInputMatrix());
        System.out.println("error:");
        MatrixMath.printMatrix(error);
        // Compute derivative
        System.out.println("Activation passed to derivative:");
        MatrixMath.printMatrix(activatedOutput.getInputMatrix());

        double[][] derivative = MatrixMath.getDerivative(layer.getAf(), activatedOutput.getInputMatrix());
        System.out.println("derivatives:");
        MatrixMath.printMatrix(derivative);
        // Compute delta
        double[][] delta = MatrixMath.hadamard(error, derivative);
        System.out.println("deltas:");
        MatrixMath.printMatrix(delta);
        // Compute gradients
        double[][] inputTransposed = MatrixMath.transpose(inputData);
        double[][] weightGradient = MatrixMath.dot(inputTransposed, delta);
        System.out.println("weightgradient:");
        MatrixMath.printMatrix(weightGradient);
        double[] biasGradient = new double[delta[0].length];
        
        for (int j = 0; j < delta[0].length; j++) {
            biasGradient[j] = delta[0][j]; // single sample
        }
        System.out.println("biasGradient:");
        System.out.println(Arrays.toString(biasGradient));
        // Apply update
        double learningRate = 0.1;
        double[][] updatedWeights = MatrixMath.subtractElementWise(
            weightData,
            MatrixMath.scalarMultiply(learningRate, weightGradient)
        );
        System.out.println("updated weights");
        MatrixMath.printMatrix(updatedWeights);
        double[] updatedBias = new double[biasData.length];
        for (int j = 0; j < biasData.length; j++) {
            updatedBias[j] = biasData[j] - learningRate * biasGradient[j];
        }

        // Assert weights changed
        assertNotEquals(weightData[0][0], updatedWeights[0][0], "Weight should be updated");
        assertNotEquals(biasData[0], updatedBias[0], "Bias should be updated");
    }
}