package matrixtests;

import static org.junit.jupiter.api.Assertions.*;
import org.junit.jupiter.api.Test;
import mlib.matrixes.*;
public class MatrixMathTest {

    @Test
    public void testDotProduct() {
        double[][] a = {{1, 2}, {3, 4}};
        double[][] b = {{5, 6}, {7, 8}};
        double[][] expected = {{19, 22}, {43, 50}};
        double[][] result = MatrixMath.dot(a, b);

        assertArrayEquals(expected, result);
    }

    @Test
    public void testHadamardProduct() {
        double[][] a = {{1, 2}, {3, 4}};
        double[][] b = {{2, 0}, {1, 2}};
        double[][] expected = {{2, 0}, {3, 8}};
        double[][] result = MatrixMath.hadamard(a, b);

        assertArrayEquals(expected, result);
    }

    @Test
    public void testScalarMultiply() {
        double[][] a = {{1, 2}, {3, 4}};
        double[][] expected = {{2, 4}, {6, 8}};
        double[][] result = MatrixMath.scalarMultiply(2.0, a);

        assertArrayEquals(expected, result);
    }

    @Test
    public void testSubtractElementWise() {
        double[][] a = {{5, 6}, {7, 8}};
        double[][] b = {{1, 2}, {3, 4}};
        double[][] expected = {{4, 4}, {4, 4}};
        double[][] result = MatrixMath.subtractElementWise(a, b);

        assertArrayEquals(expected, result);
    }

    @Test
    public void testTranspose() {
        double[][] a = {{1, 2, 3}, {4, 5, 6}};
        double[][] expected = {{1, 4}, {2, 5}, {3, 6}};
        double[][] result = MatrixMath.transpose(a);

        assertArrayEquals(expected, result);
    }

    @Test
    public void testElementWiseSquare() {
        double[][] a = {{2, -3}, {4, 0}};
        double[][] expected = {{4, 9}, {16, 0}};
        double[][] result = MatrixMath.elementWiseSquare(a);

        assertArrayEquals(expected, result);
    }

    @Test
    public void testSum() {
        double[][] a = {{1, 2}, {3, 4}};
        double expected = 10.0;
        double result = MatrixMath.sum(a);

        assertEquals(expected, result);
    }

    @Test
    public void testAddElements() {
        double[][] a = {{1, 2}, {3, 4}};
        double[] b = {10, 20};
        double[][] expected = {{11, 22}, {13, 24}};
        double[][] result = MatrixMath.addElements(a, b);

        assertArrayEquals(expected, result);
    }
}
