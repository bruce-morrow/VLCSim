import numpy as np


class LED:
    """
    A class used to represent an Light Emitting Diode (LED).

    Attributes
    ----------
    power : float
        The LED power in Watts
    power_half_angle : float
        The angle of emission of half the power in radians
    m1 : float
        The Lambert's mode number
    """

    def __init__(self, power: float, power_half_angle: float, wavelength: float):
        """
        Parameters
        ----------
        power : float
            The LED power in Watts
        power_half_angle : float
            The angle of emission of half the power in radians. Must be between 0 and pi/2
        """
        if power < 0.0:
            raise ValueError("The LED power must be non negative.")
        if power_half_angle < 0.0 or power_half_angle > np.pi/2:
            raise ValueError("The LED power half angle must be between 0 and pi/2 radians.")
        if wavelength <= 0.0:
            raise ValueError("The LED wavelength must strictly positive.")
        self._power = power
        self._power_half_angle = power_half_angle
        self._wavelength = wavelength

        # calculates the Lambert's mode number as described on equation 3.10 of "GHASSEMLOOY, Zabih; POPOOLA, Wasiu;
        # RAJBHANDARI, Sujan. Optical wireless communications: system and channel modelling with Matlab®. CRC press,
        # 2019."
        self._m1 = -np.log(2.0) / np.log(np.cos(power_half_angle))

    @property
    def power(self) -> float:
        return self._power

    @property
    def power_half_angle(self) -> float:
        return self._power_half_angle

    @property
    def wavelength(self):
        return self._wavelength

    @property
    def m1(self) -> float:
        return self._m1
