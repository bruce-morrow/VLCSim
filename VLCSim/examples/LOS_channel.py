from VLCSim.devices import LED, PhotoDiode
from VLCSim.channels import LOSChannel
from VLCSim.systems import ReceiverOnPlaneSystem
from VLCSim.utils import plot_utils

import numpy as np


def main():
    led = LED(
        power=65e-3,
        power_half_angle=np.deg2rad(30.0)
    )
    photo_diode = PhotoDiode(
        area=7.45e-6,
        field_of_view=np.deg2rad(60.0),
        transmittance=1.0,
    )
    channel = LOSChannel()
    txs1 = {led: [(0.0, 0.0, 5e-3, 0.0)]}
    txs2 = {led: [(-4e-3, -4e-3, 5e-3, 0.0),
                  (-4e-3, 4e-3, 5e-3, 0.0),
                  (4e-3, -4e-3, 5e-3, 0.0),
                  (4e-3, 4e-3, 5e-3, 0.0)]}
    system = ReceiverOnPlaneSystem(
        ul_corner=(-10e-3, 10e-3),
        lr_corner=(10e-3, -10e-3),
        num_points_axis=(100, 100),
        ch=channel,
        rx=photo_diode,
        txs=txs2
    )

    rx_powers = system.calculate_received_power()
    x_matrix = system.plane_points[:, :, 0]
    y_matrix = system.plane_points[:, :, 1]

    plot_utils.plot_3d_and_top_views(
        x_matrix=x_matrix,
        y_matrix=y_matrix,
        z_matrix=rx_powers,
        labels=("x coordinates (m)", "y coordinates (m)", "received power (W)"),
        title="Received power over plane",
        db=False
    )


if __name__ == '__main__':
    main()
