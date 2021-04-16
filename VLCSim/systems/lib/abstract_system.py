import abc
import numpy as np
from typing import Union

from channels import AbstractChannel


class AbstractSystem(abc.ABC):
    """
    An abstract class to represent optical channels.

    Attributes
    ----------
    transmitters : dict
        Dictionary where the keys are transmitter objects and the values are lists of tuples, where each tuple has four
        elements. The first three are the x, y, and z coordinates in meters and the forth is the angle of maximum
        emission angle in radians in relation ot the normal line
    receiver
        The receiver model object
    channel : AbstractChannel
        The optical channel model between transmitter(s) and receiver object

    Methods
    -------
    calculate_received_power(self, *args)
        Calculates received power from the transmitter(s) in Watts.
    """

    def __init__(self, ch: AbstractChannel, rx, txs: dict):
        """
        Parameters
        ----------
        ch : AbstractChannel
            The optical channel model between transmitter(s) and receiver object
        rx
            The receiver model object
        txs : dict
            Dictionary where the keys are transmitter objects and the values are lists of tuples, where each tuple has
            four elements. The first three are the x, y, and z coordinates in meters and the forth is the angle of
            maximum emission angle in radians in relation ot the normal line
        """
        self._transmitters = txs
        self._receiver = rx
        self._channel = ch

    @property
    def transmitters(self) -> dict:
        return self._transmitters

    @property
    def receiver(self):
        return self._receiver

    @property
    def channel(self) -> AbstractChannel:
        return self._channel

    @abc.abstractmethod
    def calculate_received_power(self, *args) -> Union[float, np.array]:
        """
        Abstract method to calculate the received power from the transmitter(s) over the channel.

        Parameters
        ----------
        args
            The all needed parameters for the calculation.

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
