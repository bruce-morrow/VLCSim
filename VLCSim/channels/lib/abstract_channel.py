import abc
from typing import Union
import numpy as np


class AbstractChannel(abc.ABC):
    """
    An abstract class to represent optical channels.

    Methods
    -------
    calculate_gain(self, *args) -> Union[float, np.array]
        Calculates the channel's optical gain.
    calculate_rx_power(self, *args) -> Union[float, np.array]
        Calculates the received power.
    """

    @abc.abstractmethod
    def calculate_gain(self, *args) -> Union[float, np.array]:
        """
        Calculates the optical gain between transmitter and receiver.

        Parameters
        ----------
        args
            The all needed parameters for the calculation.

        Returns
        -------
        Union[float, np.array]
            The channel's optical gain value(s)

        Raises
        ------
        NotImplementError
            If the method is not overwritten.
        """

        raise NotImplementedError

    @abc.abstractmethod
    def calculate_rx_power(self, *args) -> Union[float, np.array]:
        """
        Abstract method to calculate the received power from the transmitter(s) over the channel.

        Parameters
        ----------
        args
            The all needed parameters for the calculation.

        Returns
        -------
        Union[float, np.array]
            The received power value(s)

        Raises
        ------
        NotImplementError
            If the method is not overwritten.
        """

        raise NotImplementedError
