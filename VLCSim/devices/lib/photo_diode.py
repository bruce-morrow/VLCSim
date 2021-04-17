import numpy as np
import scipy.constants as const
from typing import Union


class PhotoDiode:
    """
    A class used to represent a photo diode.

    Attributes
    ----------
    area : float
        The surface area, measured in m², which receives the light. Must be non negative
    field_of_view : float
        The field of view (FOV), i.e. the maximum incidence angle measured in radians. Must be between 0 and pi/2
    external_quantum_efficiency : float
        The external quantum efficiency, i.e. the number of excitons generated over the number of received photons.
        Must be between 0 and 1, inclusive
    transmittance : float
        The light transmittance value on the surface between 0 and 1 (dimensionless). Must be between 0 and 1
    gain : float
        The internal gain. For PIN photodiodes it is equal to 1 (default value), for avalanche photodiodes it must be
        greater than 1.
    optical_concentrator_refractive_index : float
        The refractive index value of the optical concentrator if is is present. Must be greater or equal to 1 (default
        None)
    excess_noise : float
        Excess noise factor on avalanche photodiodes due to the multiplication process, must be at least 1. On PIN
        photodiodes it is equal to 1 (default value)


    Methods
    -------
    get_optical_gain(psi: Union[float, np.array])
        Returns the optical concentrator gain (dimensionless) as a function of incidence angle
    get_responsivity(self, wavelength: Union[float, np.array])
        Calculates the responsivity in Ampere/Watt as a function of received wavelength
    get_photocurrent(self, wavelength: Union[float, np.array], power: Union[float, np.array], is_outer: bool = False)
        Calculates the generated photocurrent, in Amperes, as a function of wavelenth and received power
    get_photon_fluctuation_noise_variance(...)
        Calculates the variance(s) of the photon fluctuation noise
    generate_photon_fluctuation_noise_value(...)
        Draws value(s) from a Poisson random variable with lambda (mean and variance) equal to the photon fluctuation
        noise variance.
    """

    def __init__(self,
                 area: float,
                 field_of_view: float,
                 external_quantum_efficiency: float = 1.0,
                 transmittance: float = 1.0,
                 gain: float = 1.0,
                 optical_concentrator_refractive_index: float = None,
                 excess_noise: float = 1.0
                 ):
        # TODO: add docstring
        if area < 0.0:
            raise ValueError("The photo diode area must be non negative.")
        if field_of_view < 0 or field_of_view > np.pi/2:
            raise ValueError("The photo diode field of view must be between 0 and pi/2 radians.")
        if external_quantum_efficiency < 0 or external_quantum_efficiency > 1:
            raise ValueError("The photo diode external quantum efficiency must be between 0 and 1.")
        if transmittance < 0.0 or transmittance > 1.0:
            raise ValueError("The photo diode transmittance must be between 0 and 1.")
        if gain < 1.0:
            raise ValueError("The photo diode gain must at least 1")
        if optical_concentrator_refractive_index is not None and optical_concentrator_refractive_index < 1.0:
            raise ValueError("The photo diode optical concentrator refractive index must be greater or equal to 1.")
        if excess_noise < 1.0:
            raise ValueError("The excess noise must be at least 1.")
        self._area = area
        self._field_of_view = field_of_view
        self._external_quantum_efficiency = external_quantum_efficiency
        self._transmittance = transmittance
        self._gain = gain
        self._optical_concentrator_refractive_index = optical_concentrator_refractive_index
        self._excess_noise = excess_noise

    @property
    def area(self) -> float:
        return self._area

    @property
    def field_of_view(self) -> float:
        return self._field_of_view

    @property
    def external_quantum_efficiency(self) -> float:
        return self._external_quantum_efficiency

    @property
    def transmittance(self) -> float:
        return self._transmittance

    @property
    def gain(self) -> float:
        return self._gain

    @property
    def has_optical_concentrator(self) -> bool:
        return self.optical_concentrator_refractive_index is not None

    @property
    def optical_concentrator_refractive_index(self) -> float:
        return self._optical_concentrator_refractive_index

    @property
    def excess_noise(self):
        return self._excess_noise

    def get_responsivity(self, wavelength: Union[float, np.array]) -> Union[float, np.array]:
        """
        Calculates the responsivity in Ampere/Watt as a function of received wavelength.

        The responsivity is calculated as described on equation 2.19 of "GHASSEMLOOY, Zabih; POPOOLA, Wasiu;
        RAJBHANDARI, Sujan. Optical wireless communications: system and channel modelling with Matlab®. CRC press,
        2019."

        Parameters
        ----------
        wavelength : Union[float, np.array]
            The value(s) of received wavelength(s) in meters

        Returns
        -------
        Union[float, np.array]
            The respective responsivity(ies) in Ampere/Watt
        """

        return self.gain * wavelength * const.e * self.external_quantum_efficiency / (const.h * const.c)

    def get_photocurrent(self, wavelength: Union[float, np.array], power: Union[float, np.array],
                         is_outer: bool = False) -> Union[float, np.array]:
        """
        Calculates the generated photocurrent, in Amperes, as a function of wavelenth and received power.

        The photocurrent can be calculated as an elementwise product, i.e. each wavelength corresponds to a power, or
        as an outer product, i.e. for each (wavelength, power) permutation. The calculation is performed as described by
        equation 2.18 of "GHASSEMLOOY, Zabih; POPOOLA, Wasiu; RAJBHANDARI, Sujan. Optical wireless communications:
        system and channel modelling with Matlab®. CRC press, 2019."

        Parameters
        ----------
        wavelength : Union[float, np.array]
            The value(s) of received wavelength(s) in meters
        power : Union[float, np.array]
            The value(s) of received power in Watts
        is_outer : bool
            Indicates if the calculation must be carried as an outer product, otherwise it will be elementwise

        Returns
        -------
        Union[float, np.array]
            The respective photocurrent(s) in Amperes
        """

        if is_outer:
            return np.outer(self.get_responsivity(wavelength), power)
        else:
            return self.get_responsivity(wavelength) * power

    def get_optical_gain(self, psi: Union[float, np.array]) -> Union[float, np.array]:
        """
        Calculates the optical concentrator gain (dimensionless) as a function of incidence angle.

        The gain is calculated as described on equation 3.13 of "GHASSEMLOOY, Zabih; POPOOLA, Wasiu; RAJBHANDARI, Sujan.
        Optical wireless communications: system and channel modelling with Matlab®. CRC press, 2019."

        Parameters
        ----------
        psi : Union[float, np.array]
            The value or a numpy array of incidence angles in radians

        Returns
        -------
        Union[float, np.array]
            The respective optical gain value(s)
        """
        if self.has_optical_concentrator:
            mults = np.logical_and(0.0 <= psi, psi <= self.field_of_view).astype(int)
            return mults * self.optical_concentrator_refractive_index**2 / (np.sin(self.field_of_view))**2
        elif isinstance(psi, float):
            return 1.0
        else:
            return np.ones(shape=psi.shape)

    def get_photon_fluctuation_noise_variance(
            self,
            wavelength: Union[float, np.array],
            average_power: Union[float, np.array],
            bandwidth: Union[float, np.array]
    ) -> Union[float, np.array]:
        """
        Calculates the variance(s) of the photon fluctuation noise.

        The value is calculated according to equation 2.35 of "GHASSEMLOOY, Zabih; POPOOLA, Wasiu; RAJBHANDARI, Sujan.
        Optical wireless communications: system and channel modelling with Matlab®. CRC press, 2019.", except for the
        fact that the gain is being squared, because the generated photocurrent is already multiplied by it once

        Parameters
        ----------
        wavelength : Union[float, np.array]
            The value(s) of received wavelength(s) in meters
        average_power : Union[float, np.array]
            The value(s) of average received power in Watts
        bandwidth : Union[float, np.array]
            The value(s) of bandwidth of the electrical filter(s) that follows the photodiode(s)

        Returns
        -------
        Union[float, np.array]
            The variance value(s)
        """

        return 2.0 * const.e * self.get_photocurrent(wavelength, average_power) * bandwidth * self.excess_noise * \
               self.gain

    def generate_photon_fluctuation_noise_value(
            self,
            wavelength: Union[float, np.array],
            average_power: Union[float, np.array],
            bandwidth: Union[float, np.array],
            num_values: int = 1
    ) -> Union[float, np.array]:
        """
        Draws value(s) from a Poisson random variable with lambda (mean and variance) equal to the photon fluctuation
        noise variance.

        Parameters
        ----------
        wavelength : Union[float, np.array]
            The value(s) of received wavelength(s) in meters
        average_power : Union[float, np.array]
            The value(s) of average received power in Watts
        bandwidth : Union[float, np.array]
            The value(s) of bandwidth of the electrical filter(s) that follows the photodiode(s)
        num_values : int
            Number of values to be drawn for each set of wavelength, average power and bandwidth (default 1)

        Returns
        -------
        Union[float, np.array]
            The drawn value(s)
        """

        lambdas = self.get_photon_fluctuation_noise_variance(wavelength, average_power, bandwidth)
        return np.random.poisson(lam=lambdas, size=(num_values, lambdas.size)).T
