import abc
import numpy as np
import scipy.constants as const
from typing import Union

from VLCSim.channels import AbstractChannel


class AbstractSystem(abc.ABC):
    """
    An abstract class to represent optical channels.

    Attributes
    ----------
    transmitters : dict
        Dictionary where the keys are transmitter objects and the values are lists of tuples. Each tuple represents the
        positioning of a transmitter of that type and is composed by two other tuples with three float values each. The
        first internal tuple holds the x, y, and z coordinates in meters and the other the coordinates in meters of a
        vector that points to maximum emission direction
    receiver
        The receiver model object
    channel : AbstractChannel
        The optical channel model between transmitter(s) and receiver object
    equivalent_load_resistance : float
        The equivalent input resistance, in Ohms, provided bu the electric circuit connected to the receiver, i.e the
        resistance that the receiver sees between its outputs. Must be strictly positive
        bandwidth : float
        The bandwidth, in Hertz, of the electric circuit connected to the output of the receiver

    Methods
    -------
    calculate_received_power(self, *args)
        Calculates received power from the transmitter(s) in Watts.
    get_thermal_noise_variance(...)
        Calculates the thermal noise variance for currents.
    generate_thermal_noise_value(...)
        Draws value(s) from a Gaussian distribution with 0 mean and variance equal to the thermal noise one.
    """

    def __init__(self, ch: AbstractChannel, rx, txs: dict, equivalent_load_resistance: float, bandwidth: float):
        """
        Parameters
        ----------
        ch : AbstractChannel
            The optical channel model between transmitter(s) and receiver object
        rx
            The receiver model object
        txs : dict
            Dictionary where the keys are transmitter objects and the values are lists of tuples. Each tuple represents
            the positioning of a transmitter of that type and is composed by two other tuples with three float values
            each. The first internal tuple holds the x, y, and z coordinates in meters and the other the coordinates in
            meters of a vector that points to maximum emission direction
        equivalent_load_resistance : float
            The equivalent input resistance, in Ohms, provided bu the electric circuit connected to the receiver, i.e.
            the resistance that the receiver sees between its outputs. Must be strictly positive
        bandwidth : float
            The bandwidth, in Hertz, of the electric circuit connected to the output of the receiver

        Raises
        ------
        ValueError
            If the equivalent load resistance is not strictly positive or if the bandwidth is negative.
        """

        if equivalent_load_resistance <= 0.0:
            raise ValueError("The receiver's equivalent load resistance must be strictly positive.")
        if bandwidth < 0.0:
            raise ValueError("The bandwidth of the electric circuit connected to the receiver's output must be "
                             "non negative.")
        self._transmitters = txs
        self._receiver = rx
        self._channel = ch
        self._equivalent_load_resistance = equivalent_load_resistance
        self._bandwidth = bandwidth

    @property
    def transmitters(self) -> dict:
        return self._transmitters

    @property
    def receiver(self):
        return self._receiver

    @property
    def channel(self) -> AbstractChannel:
        return self._channel

    @property
    def equivalent_load_resistance(self):
        return self._equivalent_load_resistance

    @property
    def bandwidth(self):
        return self._bandwidth

    @abc.abstractmethod
    def calculate_received_power(self, *args) -> Union[float, np.array]:
        """
        Abstract method to calculate the received power from the transmitter(s) over the channel.

        Parameters
        ----------
        args
            All the needed parameters for the calculation.

        Returns
        -------
        Union[float, np.array]
            The received power value(s) in Watts

        Raises
        ------
        NotImplementError
            If the method is not overwritten.
        """

        raise NotImplementedError

    def get_thermal_noise_variance(self, temperature: Union[float, np.array]) -> Union[float, np.array]:
        """
        Calculates the thermal noise variance for currents.

        The calculation is performed according to equation 2.45 of "GHASSEMLOOY, Zabih; POPOOLA, Wasiu; RAJBHANDARI,
        Sujan. Optical wireless communications: system and channel modelling with Matlab®. CRC press, 2019."

        Parameters
        ----------
        temperature : float
            The system's operating temperature(s) in Kelvin

        Returns
        -------
            The thermal noise variance value(s)
        """

        # TODO: see what FN means on Guilherme's code
        return 4.0 * const.k * temperature * self.bandwidth / self.equivalent_load_resistance

    def generate_thermal_noise_value(self, temperature: Union[float, np.array],
                                     num_values: int = 1) -> Union[float, np.array]:
        """
        Draws value(s) from a Gaussian distribution with 0 mean and variance equal to the thermal noise one.

        Parameters
        ----------
        temperature : float
            The system's operating temperature(s) in Kelvin
        num_values : int
            Number of values to be drawn for each temperature (default 1)

        Returns
        -------
            The thermal noise value(s) in Amperes
        """

        size = 1
        if isinstance(temperature, np.ndarray):
            size = temperature.size
        return np.random.normal(loc=0.0,
                                scale=np.sqrt(self.get_thermal_noise_variance(temperature)),
                                size=(num_values, size)).T

    def get_snr(self, temperature: Union[float, np.array], *args) -> Union[float, np.array]:
        """
        Calculates the expected Signal to Noise Ratio (SNR) in decibels.

        The calculation is carried out according to equation 2.52 of GHASSEMLOOY, Zabih; POPOOLA, Wasiu; RAJBHANDARI,
        Sujan. Optical wireless communications: system and channel modelling with Matlab®. CRC press, 2019."

        Parameters
        ----------
        temperature : float
            The system's operating temperature(s) in Kelvin

        args
            All the needed parameters for the received power calculation.

        Returns
        -------
        Union[float, np.array]
            The SNR value(s) in decibels
        """

        # TODO: this is considering that all of transmitters have the same wavelength. Later generalize for different
        #  wavelengths
        wavelength = list(self.transmitters.keys())[0].wavelength
        average_power = self.calculate_received_power(*args)
        photocurrent = self.receiver.get_photocurrent(wavelength, average_power)
        quantum_noise_var = self.receiver.get_photon_fluctuation_noise_variance(wavelength, average_power,
                                                                                self.bandwidth)
        thermal_noise_var = self.get_thermal_noise_variance(temperature)
        snr = photocurrent**2 / (quantum_noise_var + thermal_noise_var)
        return 10 * np.log10(snr)

    def get_ber(self, temperature: Union[float, np.array], *args) -> Union[float, np.array]:
        """
        Calculates the expected Bit Error Rate (BER) (dimensionless).

        Parameters
        ----------
        temperature : float
            The system's operating temperature(s) in Kelvin

        args
            All the needed parameters for the received power calculation.

        Returns
        -------
        Union[float, np.array]
            The BER value(s)
        """

        # TODO: it seems that the BER equation is dependent on the type of modulation used, Check with Guilherme where
        #  did the formula he used come from and which type of modulation it is meant for. Later generalize for
        #  different modulation schemes
        q = np.sqrt(self.get_snr(temperature, *args))
        return np.exp(-(q**2)/2) / (np.sqrt(2*np.pi) * q)
