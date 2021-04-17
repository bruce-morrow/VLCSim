from VLCSim.channels import AbstractChannel
from VLCSim.devices import LED, PhotoDiode

from typing import Union
import numpy as np


class LOSChannel(AbstractChannel):
    """
    A class to represent a Line of Sight (LOS) optical channel.

    Methods
    -------
    calculate_gain(...)
        Calculates the optical gain between transmitter and receiver
    calculate_rx_power(...)
        Calculates the received power from the transmitter(s) over the channel
    """

    def calculate_gain(
            self,
            led: LED,
            pd: PhotoDiode,
            phi: Union[float, np.array],
            psi: Union[float, np.array],
            distance: Union[float, np.array]
    ) -> Union[float, np.array]:
        """
        Calculates the optical gain between transmitter and receiver.

        The gain is calculated as described by equation 3.15 of "GHASSEMLOOY, Zabih; POPOOLA, Wasiu; RAJBHANDARI, Sujan.
        Optical wireless communications: system and channel modelling with Matlab®. CRC press, 2019."

        Parameters
        ----------
        led : LED
            The LED object to be used as transmitter
        pd : PhotoDiode
            The photo diode object to be used as receiver
        phi : Union[float, np.array]
            The value(s) of transmission angle measured in radians
        psi : Union[float, np.array]
            The values(s) of incidence angle on the receiver, measured in radians
        distance : Union[float, np.array]
            The distance(s) between transmitter and receiver, measured in meters

        Returns
        -------
        Union[float, np.array]
            The optical gain value(s) between transmitter and receiver (dimensionless)

        Raises
        ------
        ValueError
            If any distance between transmitter and receiver is negative or if the phi, psi and distance parameter do
            not have the same type and shape
        """

        if np.any(distance < 0):
            raise ValueError("The distance must be non negative.")
        if not (type(phi) == type(psi) == type(distance)):
            raise ValueError("phi, psi and distance mast have the same type.")
        if type(phi) == np.array and not (phi.shape == psi.shape == distance.shape):
            raise ValueError("phi, psi and distance mast have the same shape.")

        # matrix of multipliers equal to 0 where the photo diode FOV is exceeded and 1 where it is not
        mults = np.logical_and(0.0 <= psi, psi <= pd.field_of_view).astype(int)
        h_los_0 = mults * pd.area * (led.m1 + 1) / (2 * np.pi * distance**2)
        h_los_0 *= (np.cos(phi))**2 * pd.transmittance * pd.get_optical_gain(psi) * np.cos(psi)
        return h_los_0

    def calculate_rx_power(
            self,
            led: LED,
            pd: PhotoDiode,
            phi: Union[float, np.array],
            psi: Union[float, np.array],
            distance: Union[float, np.array]
    ) -> Union[float, np.array]:
        """
        Calculates the received power from the transmitter(s) over the channel.

        The power is calculated as described by equation 3.16 of "GHASSEMLOOY, Zabih; POPOOLA, Wasiu; RAJBHANDARI,
        Sujan. Optical wireless communications: system and channel modelling with Matlab®. CRC press, 2019."

        Parameters
        ----------
        led : LED
            The LED object to be used as transmitter
        pd : PhotoDiode
            The photo diode object to be used as receiver
        phi : Union[float, np.array]
            The value(s) of transmission angle measured in radians
        psi : Union[float, np.array]
            The values(s) of incidence angle on the receiver, measured in radians
        distance : Union[float, np.array]
            The distance(s) between transmitter and receiver, measured in meters

        Returns
        -------
        Union[float, np.array]
            The received power value(s) in Watts
        """

        return led.power * self.calculate_gain(led, pd, phi, psi, distance)
