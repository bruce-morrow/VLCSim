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
            power_half_angle=np.deg2rad(45.0),
            wavelength=680e-9
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
            txs={led: [(0.0, 0.0, 5.0, 0.0)]},
            equivalent_load_resistance=65.4e3,
            bandwidth=4.5e6
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

    def test_equivalent_load_resistance(self):
        self.assertEqual(self._sys.equivalent_load_resistance, 65.4e3, "The receiver's equivalent load resistance must "
                                                                       "be 65.4e3 ohms.")

    def test_bandwidth(self):
        self.assertEqual(self._sys.bandwidth, 4.5e6, "The bandwidth of the electric circuit connected to the "
                                                     "receiver's output must be 4.5e6 Hz.")

    def test_get_thermal_noise_variance(self):
        self.assertAlmostEqual(self._sys.get_thermal_noise_variance(300.0), 1.1399854128440366e-18, places=25,
                               msg="The thermal noise variance must be approximately 1.1399854128440366e-18 at 300K.")
        np.testing.assert_array_almost_equal(self._sys.get_thermal_noise_variance(np.array([300.0, 500.0])),
                                             np.array([1.1399854128440366e-18, 1.8999756880733945e-18]),
                                             decimal=25,
                                             err_msg="The output for an array as input must also be an array.")

    def test_generate_thermal_noise_value(self):
        self._sys.generate_thermal_noise_value(300.0)
        self.assertEqual(self._sys.generate_thermal_noise_value(temperature=np.array([300.0, 500.0]),
                                                                num_values=10).shape,
                         (2, 10),
                         "The output for an array as input must also be an array.")

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

    def test_get_snr(self):
        values = self._sys.get_snr(temperature=300.0)
        expected = np.array([[48.18213243, 55.19141415, 48.18213243],
                             [55.19141415, 64.83355694, 55.19141415],
                             [48.18213243, 55.19141415, 48.18213243]])
        np.testing.assert_array_almost_equal(values, expected)

    def test_get_ber(self):
        values = self._sys.get_ber(temperature=300.0)
        expected = np.array([[1.98084246e-12, 5.56302483e-14, 1.98084246e-12],
                             [5.56302483e-14, 4.13601524e-16, 5.56302483e-14],
                             [1.98084246e-12, 5.56302483e-14, 1.98084246e-12]])
        np.testing.assert_array_almost_equal(values, expected, decimal=20)


if __name__ == '__main__':
    unittest.main()
