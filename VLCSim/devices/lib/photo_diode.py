import numpy as np
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
    transmittance : float
        The light transmittance value on the surface between 0 and 1 (dimensionless). Must be between 0 and 1
    optical_concentrator_refractive_index : float
        The refractive index value of the optical concentrator if is is present. Must be greater or equal to 1 (default
        None)

    Methods
    -------
    get_optical_gain(psi: nd.array)
        Returns the optical concentrator gain (dimensionless)
    """

    def __init__(self,
                 area: float,
                 field_of_view: float,
                 transmittance: float = 1.0,
                 optical_concentrator_refractive_index: float = None
                 ):
        if area < 0.0:
            raise ValueError("The photo diode area must be non negative.")
        if field_of_view < 0 or field_of_view > np.pi / 2:
            raise ValueError("The photo diode field of view must be between 0 and pi/2 radians.")
        if transmittance < 0.0 or transmittance > 1.0:
            raise ValueError("The photo diode transmittance must be between 0 and 1.")
        if optical_concentrator_refractive_index is not None and optical_concentrator_refractive_index < 1.0:
            raise ValueError("The photo diode optical concentrator refractive index must be greater or equal to 1.")
        self._area = area
        self._field_of_view = field_of_view
        self._transmittance = transmittance
        self._optical_concentrator_refractive_index = optical_concentrator_refractive_index

    @property
    def area(self) -> float:
        return self._area

    @property
    def field_of_view(self) -> float:
        return self._field_of_view

    @property
    def transmittance(self) -> float:
        return self._transmittance

    @property
    def has_optical_concentrator(self) -> bool:
        return self.optical_concentrator_refractive_index is not None

    @property
    def optical_concentrator_refractive_index(self) -> float:
        return self._optical_concentrator_refractive_index

    def get_optical_gain(self, psi: Union[float, np.array]) -> Union[float, np.array]:
        """
        Calculates the optical concentrator gain (dimensionless).

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
