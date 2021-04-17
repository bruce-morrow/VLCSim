import unittest
import numpy as np

from VLCSim.systems import ReceiverOnPlaneSystem
from VLCSim.channels import LOSChannel
from VLCSim.devices import LED, PhotoDiode


class TestReceiverOnPlaneSystem(unittest.TestCase):

    def setUp(self) -> None:
        ch = LOSChannel()
        led = LED(
            power=65.0,
            power_half_angle=np.deg2rad(45.0)
        )
        pd = PhotoDiode(
            area=7.45e-6,
            field_of_view=np.deg2rad(60.0)
        )
        self._sys = ReceiverOnPlaneSystem(
            ul_corner=(-5.0, 5.0),
            lr_corner=(5.0, -5.0),
            num_points_axis=(3, 3),
            ch=ch,
            rx=pd,
            txs={led: [(0.0, 0.0, 5.0, 0.0)]}
        )

    def test_upper_left_corner(self):
        self.assertEqual(self._sys.upper_left_corner, (-5.0, 5.0), "The upper left corner coordinates must be (-5, 5).")

    def test_lower_right_corner(self):
        self.assertEqual(self._sys.lower_right_corner, (5.0, -5.0), "The lower right coordinates must be (5, -5).")

    def test_number_of_divisions(self):
        self.assertEqual(self._sys.num_points_axis, (3, 3), "The number of divisions on the x and y axis must be "
                                                            "(3, 3)")

    def test_plane_points(self):
        expected = np.array([[[-5.0, -5.0, 0.0], [-5.0, 0.0, 0.0], [-5.0, 5.0, 0.0]],
                             [[0.0, -5.0, 0.0], [0.0, 0.0, 0.0], [0.0, 5.0, 0.0]],
                             [[5.0, -5.0, 0.0], [5.0, 0.0, 0.0], [5.0, 5.0, 0.0]]])
        np.testing.assert_array_equal(self._sys.plane_points, expected)

    def test__get_point2plane_info(self):
        point = np.array([0.0, 0.0, 5])
        point_angle = 0.0
        expected_distances = np.array([[8.660254037844386, 7.071067811865476, 8.660254037844386],
                                       [7.071067811865477, 5.000000000000000, 7.071067811865476],
                                       [8.660254037844386, 7.071067811865476, 8.660254037844386]])
        expected_ang = np.array([[0.9553166181245093, np.pi/4, 0.9553166181245093],
                                 [np.pi/4, 0.00000, np.pi/4],
                                 [0.9553166181245093, np.pi/4, 0.9553166181245093]])
        distances, phi, psi = self._sys._get_point2plane_info(point_arr=point, point_angle=point_angle)
        np.testing.assert_array_almost_equal(distances, expected_distances)
        np.testing.assert_array_almost_equal(phi, expected_ang)
        np.testing.assert_array_almost_equal(psi, expected_ang)
        # TODO: test when point angle is different than 0

    def test_calculate_received_power(self):
        expected_powers = np.array([[5.93291150e-07, 1.63491816e-06, 5.93291150e-07],
                                    [1.63491816e-06, 9.24849374e-06, 1.63491816e-06],
                                    [5.93291150e-07, 1.63491816e-06, 5.93291150e-07]])
        np.testing.assert_array_almost_equal(self._sys.calculate_received_power(), expected_powers)
        # TODO: test when point angle is different than 0


if __name__ == '__main__':
    unittest.main()
