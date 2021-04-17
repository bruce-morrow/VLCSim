import unittest
import numpy as np

from VLCSim.utils import math_utils


class TestMathUtils(unittest.TestCase):

    def test_get_mesh_grid(self):

        # test passing number of points
        expected_x = np.array([[-5.0, 0.0, 5.0],
                               [-5.0, 0.0, 5.0],
                               [-5.0, 0.0, 5.0]])
        expected_y = np.array([[-5.0, -5.0, -5.0],
                               [0.0, 0.0, 0.0],
                               [5.0, 5.0, 5.0]])
        expected_mesh = np.array([[[-5.0, -5.0], [-5.0, 0.0], [-5.0, 5.0]],
                                  [[0.0, -5.0], [0.0, 0.0], [0.0, 5.0]],
                                  [[5.0, -5.0], [5.0, 0.0], [5.0, 5.0]]])
        x, y, mesh = math_utils.get_mesh_grid(
            x_limits=(-5.0, 5.0),
            y_limits=(-5.0, 5.0),
            x_num=3,
            y_num=3
        )
        np.testing.assert_array_equal(x, expected_x)
        np.testing.assert_array_equal(y, expected_y)
        np.testing.assert_array_equal(mesh, expected_mesh)

        # test passing step value
        x, y, mesh = math_utils.get_mesh_grid(
            x_limits=(-5.0, 5.0),
            y_limits=(-5.0, 5.0),
            x_step=5.0,
            y_step=5.0
        )
        np.testing.assert_array_equal(x, expected_x)
        np.testing.assert_array_equal(y, expected_y)
        np.testing.assert_array_equal(mesh, expected_mesh)

        # test the exception rising when neither the number of points nor the step value for the x axis are provided
        with self.assertRaises(ValueError):
            math_utils.get_mesh_grid(
                x_limits=(-5.0, 5.0),
                y_limits=(-5.0, 5.0),
                y_step=5.0
            )

        # test the exception rising when neither the number of points nor the step value for the y axis are provided
        with self.assertRaises(ValueError):
            math_utils.get_mesh_grid(
                x_limits=(-5.0, 5.0),
                y_limits=(-5.0, 5.0),
                x_step=5.0
            )


if __name__ == '__main__':
    unittest.main()
