from VLCSim.systems import AbstractSystem
from VLCSim.channels import LOSChannel
from VLCSim.utils import math_utils

from typing import Tuple
import numpy as np


class ReceiverOnPlaneSystem(AbstractSystem):
    """
    A class to represent VLC systems where the receiver is over a horizontal plane and teh transmitter(s) is(are) over
    it pointing down.

    Attributes
    ----------
    upper_left_corner : Tuple[float, float]
        The plane's upper left x and y coordinates, respectively, in meters
    lower_right_corner : Tuple[float, float]
        The plane's lower right corner x and y coordinates , respectively, in meters
    number_of_divisions : Tuple[int, int]
        The number of divisions in the x and y axis, respectively, to define a mesh grid
    plane_points : nd.array
        A matrix containing the x, y and z coordinates of the plane points

    Methods
    -------
    calculate_received_power(self) -> np.array
        Calculates the received power from the transmitter(s) over the channel.
    """

    def __init__(self, ul_corner: Tuple[float, float], lr_corner: Tuple[float, float], num_points_axis: Tuple[int, int],
                 ch: LOSChannel, rx, txs: dict, equivalent_load_resistance: float, bandwidth: float):
        """
        Parameters
        ----------
        ul_corner : Tuple[float, float]
            The plane's upper left x and y coordinates, respectively, in meters
        lr_corner : Tuple[float, float]
            The plane's lower right corner x and y coordinates , respectively, in meters
        num_points_axis : Tuple[int, int]
            The number of divisions in the x and y axis, respectively, to define a mesh grid
        ch : AbstractChanel
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
        """
        super().__init__(ch, rx, txs, equivalent_load_resistance, bandwidth)
        self._upper_left_corner = ul_corner
        self._lower_right_corner = lr_corner
        self._num_points_axis = num_points_axis
        _, _, plane_points = math_utils.get_mesh_grid(
            x_limits=(ul_corner[0], lr_corner[0]),
            y_limits=(lr_corner[1], ul_corner[1]),
            x_num=num_points_axis[0],
            y_num=num_points_axis[1]
        )

        # add z=0 coordinate to the plane points matrix
        self._plane_points = np.zeros((plane_points.shape[0], plane_points.shape[1], plane_points.shape[2] + 1))
        self._plane_points[:, :, :-1] = plane_points

    @property
    def upper_left_corner(self) -> Tuple[float, float]:
        return self._upper_left_corner

    @property
    def lower_right_corner(self) -> Tuple[float, float]:
        return self._lower_right_corner

    @property
    def num_points_axis(self) -> Tuple[int, int]:
        return self._num_points_axis

    @property
    def plane_points(self) -> np.array:
        return self._plane_points

    def calculate_received_power(self) -> np.array:
        """
        Calculates the received power from the transmitter(s) over the channel.

        Returns
        -------
        Union[float, np.array]
            The received power value(s) in Watts
        """

        rx_power = np.zeros(shape=self.plane_points.shape[:-1])
        for tx, positions in self.transmitters.items():
            for (point_coords, point_direc) in positions:
                dists, phi, psi = self._get_point2plane_info(point_coords, point_direc)
                # add up the received power matrix emitted from each transmitter
                rx_power += self.channel.calculate_rx_power(tx, self.receiver, phi, psi, dists)
        return rx_power

    def _get_point2plane_info(self, point_coords: np.array, point_direc: np.array) -> \
            Tuple[np.array, np.array, np.array]:
        """
        Calculates the distance, transmission and incidence angles from a point to every point on the plane.

        Parameters
        ----------
        point_coords : np.array
            An array with the point's x, y and z coordinates in meters
        point_direc : np.array
            The point's maximum power transmission angle, in radians, in relation to the plane's normal vector

        Returns
        -------
        Tuple[np.array, np.array, np.array]
            The distances, emission and incidence angles matrices, respectively
        """

        dist_vecs = self.plane_points - point_coords
        norm_vec = np.array([point_coords[0], point_coords[1], 0]) - point_coords
        norm_vec /= np.linalg.norm(norm_vec)
        distances = np.linalg.norm(dist_vecs, axis=2)
        psi = np.arccos(np.dot(dist_vecs, norm_vec) / distances)
        phi = np.arccos(np.dot(dist_vecs, point_direc) / (distances * np.linalg.norm(point_direc)))
        return distances, phi, psi
