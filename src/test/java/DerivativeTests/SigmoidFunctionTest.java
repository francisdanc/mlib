package DerivativeTests;

import static org.junit.jupiter.api.Assertions.*;
import org.junit.jupiter.api.Test;
import mlib.network.activations.*;
import mlib.matrixes.*;
public class SigmoidFunctionTest {

    @Test
    public void testSigmoidDerivativeFromActivation() {
        ActivationFunction sigmoid = new sigmoid();

        // Known activation values and expected derivatives
        double[][] activations = {
            {0.0, 0.5, 1.0},
            {0.25, 0.75, 0.9}
        };

        double[][] expectedDerivatives = {
            {0.0, 0.25, 0.0},
            {0.1875, 0.1875, 0.09}
        };

        double[][] actualDerivatives = MatrixMath.getDerivative(sigmoid, activations);

        for (int i = 0; i < activations.length; i++) {
            for (int j = 0; j < activations[0].length; j++) {
                assertEquals(expectedDerivatives[i][j], actualDerivatives[i][j], 1e-6,
                    "Mismatch at [" + i + "][" + j + "]");
            }
        }
    }
}