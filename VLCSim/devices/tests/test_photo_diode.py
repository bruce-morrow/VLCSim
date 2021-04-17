import unittest
import numpy as np

from VLCSim.devices import PhotoDiode


class TestPhotoDiode(unittest.TestCase):

    def setUp(self) -> None:
        self._pd1 = PhotoDiode(
            area=7.45e-6,
            field_of_view=np.deg2rad(60.0),
            transmittance=0.8,
            external_quantum_efficiency=0.75
        )
        self._pd2 = PhotoDiode(
            area=3e-6,
            field_of_view=np.deg2rad(30.0),
            gain=2.0,
            optical_concentrator_refractive_index=1.2,
            excess_noise=1.5
        )

    def test_area(self):
        self.assertEqual(self._pd1.area, 7.45e-6, "The photo diode 1 area must be 7.45 mm².")

    def test_field_of_view(self):
        self.assertEqual(self._pd1.field_of_view, np.deg2rad(60.0), "The photo diode 1 field of view must be pi/3 rad.")

    def test_external_quantum_efficiency(self):
        self.assertEqual(self._pd1.external_quantum_efficiency, 0.75, "The photo diode 1 external quantum efficiency "
                                                                      "must be 0.75.")
        self.assertEqual(self._pd2.external_quantum_efficiency, 1.0, "The photo diode 2 external quantum efficiency "
                                                                     "must be 1.")

    def test_transmittance(self):
        self.assertEqual(self._pd1.transmittance, 0.8, "The photo diode 1 transmittance must be 0.8.")
        self.assertEqual(self._pd2.transmittance, 1.0, "The photo diode 2 transmittance must be 1.0.")

    def test_has_optical_concentrator(self):
        self.assertEqual(self._pd1.has_optical_concentrator, False, "The photo diode 1 must not have an optical "
                                                                    "concentrator.")
        self.assertEqual(self._pd2.has_optical_concentrator, True, "The photo diode 2 must not have an optical "
                                                                   "concentrator.")

    def test_optical_concentrator_refractive_index(self):
        self.assertEqual(self._pd1.optical_concentrator_refractive_index, None, "The photo diode 1 optical "
                                                                                "concentrator refractive index must be "
                                                                                "None")
        self.assertEqual(self._pd2.optical_concentrator_refractive_index, 1.2, "The photo diode 2 optical concentrator "
                                                                               "refractive index must be 1.2.")

    def test_excess_noise(self):
        self.assertEqual(self._pd1.excess_noise, 1.0, "The photo diode 1 excess noise must be 1.")
        self.assertEqual(self._pd2.excess_noise, 1.5, "The photo diode 2 excess noise must be 1.5.")

    def test_get_responsivity(self):
        self.assertAlmostEqual(self._pd1.get_responsivity(680e-9), 0.41134274080480976,
                               msg="The photo diode 2 responsivity must be approximately 0.41134274080480976.")
        np.testing.assert_array_almost_equal(self._pd2.get_responsivity(np.array([500e-9, 600e-9])),
                                             np.array([0.80655439, 0.96786527]),
                                             err_msg="The output of an array as input must also be an array.")

    def test_get_photocurrent(self):
        self.assertAlmostEqual(self._pd1.get_photocurrent(wavelength=680e-9, power=15e-3), 0.006170141112072146)
        np.testing.assert_array_almost_equal(self._pd2.get_photocurrent(wavelength=np.array([500e-9, 600e-9]),
                                                                        power=np.array([15e-3, 25e-3])),
                                             np.array([0.01209832, 0.02419663]),
                                             err_msg="The output of arrays as input must also be an array when the "
                                                     "product option is elementwise.")
        np.testing.assert_array_almost_equal(self._pd2.get_photocurrent(wavelength=np.array([500e-9, 600e-9]),
                                                                        power=np.array([15e-3, 25e-3]),
                                                                        is_outer=True),
                                             np.array([[0.01209832, 0.02016386],
                                                       [0.01451798, 0.02419664]]),
                                             err_msg="The output of arrays as input must also be a matrix when the "
                                                     "product option is outer.")

    def test_get_optical_gain(self):
        self.assertEqual(self._pd1.get_optical_gain(np.deg2rad(30.0)), 1.0, "The photo diode 1 optical gain must be 1.")
        np.testing.assert_array_equal(
            self._pd1.get_optical_gain(np.deg2rad([30.0, 60.0, 80.0])),
            np.array([1.0, 1.0, 1.0]),
            err_msg="The output for an array as input must be an array. The photo diode 1 optical gain must be 1 "
                    "everywhere."
        )
        self.assertAlmostEqual(self._pd2.get_optical_gain(np.deg2rad(20.0)), 5.76,
                               msg="The photo diode 2 optical gain inside the field of view must be 5.76.")
        np.testing.assert_array_almost_equal(
            self._pd2.get_optical_gain(np.deg2rad([15.0, 30.0, 45.0])),
            np.array([5.76, 5.76, 0.0]),
            err_msg="The output for an array as input must be an array. The photo diode 2 optical gain inside and at "
                    "the FOV must be 5.76 and outside of it must be 0."
        )

    def test_get_photon_fluctuation_noise_variance(self):
        self.assertAlmostEqual(self._pd1.get_photon_fluctuation_noise_variance(680e-9, 15e-3, 1e6),
                               1.99713116637e-15,
                               places=16,
                               msg="The photo diode 1 photon fluctuation noise variance must be approximately "
                                   "1.99713116637e-15.")
        np.testing.assert_array_almost_equal(
            self._pd2.get_photon_fluctuation_noise_variance(wavelength=np.array([500e-9, 600e-9]),
                                                            average_power=np.array([15e-3, 25e-3]),
                                                            bandwidth=np.array([1e6, 2e6])),
            np.array([1.16301874e-14, 4.65207302e-14]),
            decimal=16,
            err_msg="The output for arrays as input must also be an array.")

    def test_generate_photon_fluctuation_noise_value(self):
        self._pd1.get_photon_fluctuation_noise_variance(680e-9, 15e-3, 1e6)
        self.assertEqual(
            self._pd2.generate_photon_fluctuation_noise_value(wavelength=np.array([500e-9, 600e-9]),
                                                              average_power=np.array([15e-3, 25e-3]),
                                                              bandwidth=np.array([1e6, 2e6]),
                                                              num_values=10).shape,
            (2, 10),
            "The generated values array shape must be (2, 10).")

    def test_value_error(self):
        with self.assertRaises(ValueError):
            PhotoDiode(-7.45e-6, np.pi/4)
        with self.assertRaises(ValueError):
            PhotoDiode(7.45e-6, -np.pi/4)
        with self.assertRaises(ValueError):
            PhotoDiode(7.45e-6, np.pi)
        with self.assertRaises(ValueError):
            PhotoDiode(7.45e-6, np.pi/4, external_quantum_efficiency=-0.75)
        with self.assertRaises(ValueError):
            PhotoDiode(7.45e-6, np.pi/4, external_quantum_efficiency=2)
        with self.assertRaises(ValueError):
            PhotoDiode(7.45e-6, np.pi/4, transmittance=-0.5)
        with self.assertRaises(ValueError):
            PhotoDiode(7.45e-6, np.pi / 4, transmittance=2)
        with self.assertRaises(ValueError):
            PhotoDiode(7.45e-6, np.pi / 4, optical_concentrator_refractive_index=0.5)
        with self.assertRaises(ValueError):
            PhotoDiode(7.45e-6, np.pi / 4, excess_noise=0.5)
        with self.assertRaises(ValueError):
            PhotoDiode(7.45e-6, np.pi / 4, gain=0.5)


if __name__ == '__main__':
    unittest.main()
