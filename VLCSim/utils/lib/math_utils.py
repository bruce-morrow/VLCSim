"""
Module containing mathematical utilities.

Functions
---------
get_mesh_grid(...) -> np.array
    Creates a mesh grid on a specified plane.
"""

import numpy as np
from typing import Tuple
from itertools import product


def get_mesh_grid(
        x_limits: Tuple[float, float],
        y_limits: Tuple[float, float],
        x_step: float = None,
        x_num: int = None,
        y_step: float = None,
        y_num: float = None
) -> np.array:
    """
    Creates a mesh grid on a specified plane.

    Either x_step or x_num parameters must be provided. If both are, than x_step will be ignored. Same follows for the
    y axis.

    Parameters
    ----------
    x_limits : Tuple[float, float]
        The minimum and maximum x coordinates values in meters
    y_limits : Tuple[float, float]
        The minimum and maximum x coordinates values in meters
    x_step : float
        The step value to generate the x coordinates points (default None)
    x_num : float
        The number of points to generate on the x axis (default None)
    y_step : float
        The step value to generate the y coordinates points (default None)
    y_num : float
        The number of points to generate on the y axis (default None)

    Returns
    -------
    *np.meshgrid(x, y)
        The x and y coordinates matrices of the generate mesh grid
    np.array
        The matrix of points' x and y coordinates, respectively
    """

    if (x_step is None and x_num is None) or \
            (x_step is not None and x_num is not None) or \
            (y_step is None and y_num is None) or \
            (y_step is not None and y_num is not None):
        raise ValueError("You must define the step or the number of points (mutual exclusive) for both dimensions.")
    if x_num is not None:
        x = np.linspace(x_limits[0], x_limits[1], num=x_num)
    else:
        x = np.arange(x_limits[0], x_limits[1] + x_step/2, step=x_step)
    if y_num is not None:
        y = np.linspace(y_limits[0], y_limits[1], num=y_num)
    else:
        y = np.arange(y_limits[0], y_limits[1] + y_step/2, step=y_step)
    return *np.meshgrid(x, y), np.array(list(product(x, y))).reshape((x.size, y.size, 2))
