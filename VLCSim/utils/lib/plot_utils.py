"""
Module containing plotting utilities.

Functions
---------
plot_3d_and_top_views(...) -> None
    Plots the 3D and top views of a given surface.
"""

import plotly.graph_objects as go
from plotly.subplots import make_subplots
from typing import Tuple
import numpy as np


def plot_3d_and_top_views(
        x_matrix: np.array,
        y_matrix: np.array,
        z_matrix: np.array,
        labels: Tuple[str, str, str],
        axis_tooltips: Tuple[str, str, str],
        axis_units: Tuple[str, str, str],
        title: str,
        db=False
) -> None:
    """
    Plots the 3D and top views of a given surface.

    Parameters
    ----------
    x_matrix : np.array
        The mesh grid x coordinates matrix
    y_matrix : np.array
        The mesh grid y coordinates matrix
    z_matrix : np.array
        The mesh grid z coordinates matrix
    labels : Tuple[str, str, str]
        The labels for the x, y and z axis respectively
    axis_tooltips : Tuple[str, str, str]
        The tooltips' variables names for the x, y and z axis respectively
    axis_units : Tuple[str, str, str]
        The measurement units for the x, y and z axis respectively
    title : str
        The plot title
    db : bool
        Indicates whether to plot the z axis in a dB scale (default False)
    """

    fig = make_subplots(
        rows=1, cols=2,
        specs=[[dict(type='surface'), dict(type='contour')]],
        column_widths=[0.5, 0.4],
        horizontal_spacing=0.1,
        subplot_titles=['3D view', 'Top view']
    )
    z_values: np.array
    if db:
        z_values = 10 * np.log10(z_matrix)
    else:
        z_values = z_matrix
    hover_template = axis_tooltips[0] + ": %{x:.3s}" + axis_units[0] + "<br>" + \
                     axis_tooltips[1] + ": %{y:.3s}" + axis_units[1] + "<br>" + \
                     axis_tooltips[2] + ": %{z:.3s}" + axis_units[2]
    fig.add_trace(
        go.Surface(
            x=x_matrix,
            y=y_matrix,
            z=z_values,
            colorscale='Plasma',
            showscale=False,
            name='3D view',
            hovertemplate=hover_template
        ),
        row=1, col=1
    )
    fig.add_trace(
        go.Contour(
            x=x_matrix.ravel(),
            y=y_matrix.ravel(),
            z=z_values.ravel(),
            contours_coloring='heatmap',
            showscale=True,
            colorscale='Plasma',
            name='Top view',
            hovertemplate=hover_template,
            colorbar=dict(title=axis_tooltips[2], exponentformat='SI', ticksuffix=axis_units[2]),
        ),
        row=1, col=2
    )
    fig.update_layout(title=title,
                      scene=dict(xaxis=dict(nticks=5, title=labels[0], showspikes=True),
                                 yaxis=dict(nticks=5, title=labels[1], showspikes=True),
                                 zaxis=dict(nticks=5, title=labels[2], showspikes=True)),
                      template='plotly_dark')
    fig.update_xaxes(row=1, col=2, title=labels[0])
    fig.update_yaxes(row=1, col=2, title=labels[1])
    fig.show()
