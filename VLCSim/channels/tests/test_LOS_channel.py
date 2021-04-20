import unittest
import numpy as np

from VLCSim.channels import LOSChannel
from VLCSim.devices import LED, PhotoDiode


class TestLOSChannel(unittest.TestCase):

    def setUp(self) -> None:
        self._ch = LOSChannel()
        self._led = LED(
            power=65e-3,
            power_half_angle=np.deg2rad(45.0),
            wavelength=620e-9
        )
        self._pd = PhotoDiode(
            area=7.45e-6,
            field_of_view=np.deg2rad(45.0),
            transmittance=0.8,
            optical_concentrator_refractive_index=1.2
        )

    def test_calculate_gain(self):

        # test single input inside the FOV range
        val1 = self._ch.calculate_gain(
            led=self._led,
            pd=self._pd,
            phi=float(np.deg2rad(15.0)),
            psi=float(np.deg2rad(30.0)),
            distance=5e-3
        )
        self.assertAlmostEqual(val1, 0.264885577883, msg="The optical gain must be approximately 0.264885577883.")

        # test single input outside the FOV range
        val2 = self._ch.calculate_gain(
            led=self._led,
            pd=self._pd,
            phi=float(np.deg2rad(15.0)),
            psi=float(np.deg2rad(60.0)),
            distance=5e-3
        )
        self.assertEqual(val2, 0.0, "The optical gain must be 0 outside of the photo diode's FOV.")

        # test multiple inputs
        val_arr = self._ch.calculate_gain(
            led=self._led,
            pd=self._pd,
            phi=np.deg2rad([15.0, 30.0, 45.0]),
            psi=np.deg2rad([30.0, 45.0, 60.0]),
            distance=np.array([5e-3, 5e-3, 5e-3])
        )
        expected = np.array([0.264885577883, 0.173854681914, 0.0])
        np.testing.assert_array_almost_equal(val_arr, expected, err_msg="If psi, phi and distance are arrays, than the "
                                                                        "output must be an array.")

        # test the exception rising for single negative distance
        with self.assertRaises(ValueError):
            self._ch.calculate_gain(
                led=self._led,
                pd=self._pd,
                phi=float(np.deg2rad(15.0)),
                psi=float(np.deg2rad(60.0)),
                distance=-5e-3
            )

        # test the exception rising for different types
        with self.assertRaises(ValueError):
            self._ch.calculate_gain(
                led=self._led,
                pd=self._pd,
                phi=float(np.deg2rad(15.0)),
                psi=float(np.deg2rad(60.0)),
                distance=np.array([5e-3, 5e-3, 5e-3])
            )

        # test the exception rising for negative distance within array
        with self.assertRaises(ValueError):
            self._ch.calculate_gain(
                led=self._led,
                pd=self._pd,
                phi=np.deg2rad([15.0, 30.0, 45.0]),
                psi=np.deg2rad([30.0, 45.0, 60.0]),
                distance=np.array([5e-3, -5e-3, 5e-3])
            )

        # test the exception rising for different array shapes
        with self.assertRaises(ValueError):
            self._ch.calculate_gain(
                led=self._led,
                pd=self._pd,
                phi=np.deg2rad([[15.0, 30.0, 45.0]]),
                psi=np.deg2rad([30.0, 45.0, 60.0]),
                distance=np.array([5e-3, 5e-3, 5e-3])
            )

    def test_calculate_rx_power(self):

        # test single input inside the FOV range
        val1 = self._ch.calculate_rx_power(
            led=self._led,
            pd=self._pd,
            phi=float(np.deg2rad(15.0)),
            psi=float(np.deg2rad(30.0)),
            distance=5e-3
        )
        self.assertAlmostEqual(val1, 1.72175625624e-2, msg="The received power must be approximately 1.72175625624e-2.")

        # test single input outside the FOV range
        val2 = self._ch.calculate_rx_power(
            led=self._led,
            pd=self._pd,
            phi=float(np.deg2rad(15.0)),
            psi=float(np.deg2rad(60.0)),
            distance=5e-3
        )
        self.assertEqual(val2, 0.0, "The received power must be 0 outside of the photo diode's FOV.")

        # test multiple inputs
        val_arr = self._ch.calculate_rx_power(
            led=self._led,
            pd=self._pd,
            phi=np.deg2rad([15.0, 30.0, 45.0]),
            psi=np.deg2rad([30.0, 45.0, 60.0]),
            distance=np.array([5e-3, 5e-3, 5e-3])
        )
        expected = np.array([1.72175625624e-2, 1.13005543244e-2, 0.0])
        np.testing.assert_array_almost_equal(val_arr, expected, err_msg="If psi, phi and distance are arrays, than the "
                                                                        "output must be an array.")


if __name__ == '__main__':
    unittest.main()
