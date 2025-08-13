package matrixtests;

import static org.junit.jupiter.api.Assertions.*;
import org.junit.jupiter.api.Test;

import mlib.matrixes.InputMatrix;
import mlib.matrixes.WeightMatrix;
import mlib.network.layers.*;
public class DenseLayerTest {

    @Test
    public void testSum() {
        // Setup: 2 samples, each with 3 features
        double[][] inputData = {
            {1.0, 2.0, 3.0},
            {4.0, 5.0, 6.0}
        };

        // Weights: 3 inputs → 2 neurons
        double[][] weightData = {
            {0.1, 0.2},
            {0.3, 0.4},
            {0.5, 0.6}
        };

        // Bias: 2 neurons
        double[] biasData = {0.5, -0.5};

        // Expected output:

        double[][] expected = {
            {2.7, 2.3},
            {5.4, 5.9}
        };

        // Create layer
        DenseLayer layer = new DenseLayer(2, 3); // 2 neurons, 3 inputs
        layer.setBias(biasData);

        // Set weights manually
        WeightMatrix wm = new WeightMatrix();
        wm.setWeights(weightData);
        layer.setWeights(wm);

        // Set inputs
        InputMatrix inputMatrix = new InputMatrix(2, 3);
        inputMatrix.setInputMatrix(inputData);
        layer.Input(inputMatrix);

        // Run Sum
        double[][] result = layer.Sum().getInputMatrix();

        // Assert
        for (int i = 0; i < expected.length; i++) {
            for (int j = 0; j < expected[0].length; j++) {
                assertEquals(expected[i][j], result[i][j], 1e-6, "Mismatch at [" + i + "][" + j + "]");
            }
        }
    }
}