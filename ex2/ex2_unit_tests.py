import unittest
import numpy as np
from logistic_regression import sigmoid


class SigmoidTester(unittest.TestCase):
    theta = np.array([1, 2])
    x = np.array([[-10, 0, 10], [-10, 0, 10]])
    expected_shape = (3,)
    expected_answer = [1*10**-13, 0.5, 1-1*10**-13]

    def test_result_shape(self):
        """ Test the function returns the right shape, and the bounds of the sigmoid function. """
        result = sigmoid(self.theta, self.x)
        self.assertEqual(result.shape, self.expected_shape)

    def test_result_values(self):
        """ Test the bounds of the result, [should be very small, 0.5, very close to 1] """
        result = sigmoid(self.theta, self.x)
        self.assertLess(result[0], self.expected_answer[0])
        self.assertEqual(result[1], self.expected_answer[1])
        self.assertGreater(result[2], self.expected_answer[2])


def main():
    unittest.main()


if __name__ == "__main__":
    main()