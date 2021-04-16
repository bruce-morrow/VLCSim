import unittest
import numpy as np

from devices import PhotoDiode


class TestPhotoDiode(unittest.TestCase):

    def setUp(self) -> None:
        self._pd1 = PhotoDiode(
            area=7.45e-6,
            field_of_view=np.deg2rad(60.0),
            transmittance=0.8
        )
        self._pd2 = PhotoDiode(
            area=3e-6,
            field_of_view=np.deg2rad(30.0),
            optical_concentrator_refractive_index=1.2
        )

    def test_area(self):
        self.assertEqual(self._pd1.area, 7.45e-6, "The photo diode 1 area must be 7.45 mm².")

    def test_field_of_view(self):
        self.assertEqual(self._pd1.field_of_view, np.deg2rad(60.0), "The photo diode 1 field of view must be pi/3 rad.")

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

    def test_value_error(self):
        with self.assertRaises(ValueError):
            PhotoDiode(-7.45e-6, np.pi/4)
        with self.assertRaises(ValueError):
            PhotoDiode(7.45e-6, -np.pi/4)
        with self.assertRaises(ValueError):
            PhotoDiode(7.45e-6, np.pi)
        with self.assertRaises(ValueError):
            PhotoDiode(7.45e-6, np.pi/4, transmittance=-0.5)
        with self.assertRaises(ValueError):
            PhotoDiode(7.45e-6, np.pi / 4, transmittance=2)
        with self.assertRaises(ValueError):
            PhotoDiode(7.45e-6, np.pi / 4, optical_concentrator_refractive_index=0.5)


if __name__ == '__main__':
    unittest.main()
