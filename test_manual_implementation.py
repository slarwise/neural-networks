#!/Library/Frameworks/Python.framework/Versions/3.7/bin/python3
import unittest
import manual_implementation
import numpy as np

class TestQ1(unittest.TestCase):

    def test_compute_current_b(self):
        W_current = np.array([[0, 1, 2], [1, 2, 3]])
        V_prev = np.array([[1, 3, 2]])
        theta_current = np.array([1, -2])
        actual = q1.compute_current_b(W_current, V_prev, theta_current)
        expected = np.array([[6, 15]])
        self.assertTrue(np.array_equal(actual, expected))
        W_current = np.array([[0, 1, 2], [1, 2, 3]])
        V_prev = np.array([[1, 3, 2], [0, 2, 1]])
        theta_current = np.array([1, -2])
        actual = q1.compute_current_b(W_current, V_prev, theta_current)
        expected = np.array([[6, 15], [3, 9]])
        self.assertTrue(np.array_equal(actual, expected))

    def test_compute_current_layer(self):
        W_current = np.array([[0, 1, 2], [1, 2, 3]])
        V_prev = np.array([[1, 3, 2], [0, 2, 1]])
        theta_current = np.array([1, -2])
        actual = q1.compute_current_layer(W_current, V_prev, theta_current)
        b_1 = np.array([[6, 15], [3, 9]])
        expected = 1 / (1+np.exp(-b_1))
        self.assertTrue(np.array_equal(actual, expected))

    def test_compute_output_error(self):
        b_current = np.array([[6, 15], [3, 9]])
        output = np.array([[0.1, 0.2], [0.2, 0.3]])
        targets = np.array([[0, 1], [1, 0]])
        expected = 1/(1+np.exp(-b_current)) * (1-1/(1+np.exp(-b_current))) \
                * (targets - output)
        actual = q1.compute_output_error(b_current, output, targets)
        self.assertTrue(np.array_equal(expected, actual))

    def test_compute_current_error(self):
        b_current = np.array([[0, 0, 2], [4, 8, 14]])
        expected = np.array([[1, 2, 3], [3, 8, 13]]) * \
                1/(1+np.exp(-b_current)) * (1 - 1/(1+np.exp(-b_current)))
        error_next = np.array([[0, 1], [2, 3]])
        W_next = np.array([[0, 1, 2], [1, 2, 3]])
        actual = q1.compute_current_error(error_next, W_next, b_current)
        self.assertTrue(np.allclose(expected, actual))

    def test_compute_weight_increment(self):
        error_current = np.array([[0, 1], [2, 3]])
        nu = 0.1
        expected = nu * np.array([[6, 8, 10], [9, 13, 17]])
        V_prev = np.array([[0, 1, 2], [3, 4, 5]])
        actual = q1.compute_weight_increment(error_current, V_prev, nu)
        self.assertTrue(np.array_equal(expected, actual))

    def test_compute_theta_increment(self):
        nu = 0.2
        error_current = np.array([[0, 1.], [2, 3]])
        actual = q1.compute_theta_increment(error_current, nu)
        expected = -nu*np.array([2, 4])
        self.assertTrue(np.array_equal(expected, actual))

    def test_classification_error(self):
        o = np.array([[0.2, 0.5, 0.3], [0.9, 0.3, 0.2]])
        t = np.array([[0, 1, 0], [1, 0, 0]])
        expected = 0
        actual = q1.classification_error(o, t)
        self.assertEqual(expected, actual)
        o = np.array([[0.2, 0.5, 0.3], [0.9, 0.3, 0.2]])
        t = np.array([[1, 0, 0], [1, 0, 0]])
        expected = 0.5
        actual = q1.classification_error(o, t)
        self.assertEqual(expected, actual)

    def test_sigmoid(self):
        b = 0
        actual = q1.sigmoid(b)
        expected = 1/2
        self.assertEqual(expected, actual)

    def test_sigmoid_prime(self):
        b = 0
        actual = q1.sigmoid_prime(b)
        expected = 1/4
        self.assertEqual(expected, actual)


if __name__ == "__main__":
    unittest.main(verbosity=0)
