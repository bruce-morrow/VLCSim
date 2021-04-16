import unittest
import numpy as np

from devices import LED


class TestLED(unittest.TestCase):

    def setUp(self) -> None:
        self._led = LED(
            power=65e-3,
            power_half_angle=np.deg2rad(45.0)
        )

    def test_power(self):
        self.assertEqual(self._led.power, 65e-3, "The LED's power must be 65 mW.")

    def test_power_half_angle(self):
        self.assertEqual(self._led.power_half_angle, np.deg2rad(45.0), "The LED's power half angle must be pi/4 rad.")

    def test_m1(self):
        self.assertAlmostEqual(self._led.m1, 2.0000000000000004,
                               msg="The LED's Lambert's mode number must be approximately 2.")

    def test_value_error(self):
        with self.assertRaises(ValueError):
            LED(-10e-3, np.pi/4)
        with self.assertRaises(ValueError):
            LED(65e-3, -np.pi/4)
        with self.assertRaises(ValueError):
            LED(65e-3, np.pi)


if __name__ == '__main__':
    unittest.main()
