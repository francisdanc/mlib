package test.tests;
import static org.junit.jupiter.api.Assertions.*;
import org.junit.jupiter.api.BeforeEach;
import org.junit.jupiter.api.Test;

import mlib.matrixes.InputMatrix;
import mlib.matrixes.WeightMatrix;
import mlib.network.layers.DenseLayer;

public class DenseLayerTest {

    private DenseLayer layer;
    private InputMatrix inputs;
    private WeightMatrix weights;

    @BeforeEach
    public void setup() {
        // Assume getHeight() = 2 for simplicity
          // Or however it's constructed

        // Set inputs: 2 batches, each with 3 input neurons
        double[][] inputData = {
            {1.0, 2.0, 3.0},
            {4.0, 5.0, 6.0}
        };
        inputs = new InputMatrix(2, 3); // batch size 2, feature size 3
        inputs.setInputMatrix(inputData);
        
        double[][] weightData = {
                {0.5, 1.0}, {0.5, 0.0}, {0.5, -1.0}};   // First output neuron
                  // Second output neuron;
        
        
        //n = 3, h = 2, b = 2
        layer = new DenseLayer(2);
        layer.setInputs(inputs);
        layer.initialiseWeights();
        layer.getWeightMatrix().setWeights(weightData);
    }

    @Test
    public void testSum() {
        InputMatrix result = layer.Sum();
        double[][] output = result.getInputMatrix();

        // Let's clarify the inputs for two batches:
        // Batch 0: {1.0, 2.0, 3.0}
        // Batch 1: {4.0, 5.0, 6.0}

        // Using weights:
        // Output neuron 0: dot([0.5, 0.5, 0.5]) with input
        // Output neuron 1: dot([1.0, 0.0, -1.0]) with input

        // Calculations:

        // Batch 0
        // output[0][0] = 1*0.5 + 2*0.5 + 3*0.5 = 3.0
        // output[0][1] = 1*1.0 + 2*0.0 + 3*(-1.0) = -2.0

        // Batch 1
        // output[1][0] = 4*0.5 + 5*0.5 + 6*0.5 = 7.5
        // output[1][1] = 4*1.0 + 5*0.0 + 6*(-1.0) = -2.0

        assertEquals(3.0, output[0][0], 1e-6);
        assertEquals(-2.0, output[0][1], 1e-6);
        assertEquals(7.5, output[1][0], 1e-6);
        assertEquals(-2.0, output[1][1], 1e-6);
    }

    @Test
    public void testThrowsIfInputsNull() {
        layer.setInputs(null);
        assertThrows(RuntimeException.class, () -> {
            layer.Sum();
        });
    }
}
