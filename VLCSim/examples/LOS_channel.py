from VLCSim.devices import LED, PhotoDiode
from VLCSim.channels import LOSChannel
from VLCSim.systems import ReceiverOnPlaneSystem
from VLCSim.utils import plot_utils

import numpy as np


def main():
    led = LED(
        power=65e-3,
        power_half_angle=np.deg2rad(30.0),
        wavelength=620e-9
    )
    photo_diode = PhotoDiode(
        area=7.45e-6,
        field_of_view=np.deg2rad(60.0),
        transmittance=1.0,
        external_quantum_efficiency=0.75
    )
    channel = LOSChannel()
    txs1 = {led: [(0.0, 0.0, 30e-2, 0.0)]}
    txs2 = {led: [(-4e-3, -4e-3, 5e-3, 0.0),
                  (-4e-3, 4e-3, 5e-3, 0.0),
                  (4e-3, -4e-3, 5e-3, 0.0),
                  (4e-3, 4e-3, 5e-3, 0.0)]}
    system = ReceiverOnPlaneSystem(
        ul_corner=(-0.5, 0.5),
        lr_corner=(0.5, -0.5),
        num_points_axis=(100, 100),
        ch=channel,
        rx=photo_diode,
        txs=txs1,
        equivalent_load_resistance=65.4e3,
        bandwidth=4.5e6
    )
    temperature = 293.0

    rx_powers = system.calculate_received_power()
    photo_current = photo_diode.get_photocurrent(led.wavelength, rx_powers)
    snr = system.get_snr(temperature)
    ber = system.get_ber(temperature)
    x_matrix = system.plane_points[:, :, 0]
    y_matrix = system.plane_points[:, :, 1]
    x_label = "x coordinates (m)"
    y_label = "y coordinates (m)"

    plot_utils.plot_3d_and_top_views(
        x_matrix=x_matrix,
        y_matrix=y_matrix,
        z_matrix=rx_powers,
        labels=(x_label, y_label, "received power (W)"),
        title="Received power over plane",
        axis_tooltips=("x", "y", "power"),
        axis_units=("m", "m", "W"),
        db=False
    )

    plot_utils.plot_3d_and_top_views(
        x_matrix=x_matrix,
        y_matrix=y_matrix,
        z_matrix=photo_current,
        labels=(x_label, y_label, "photocurrent (A)"),
        title="Generated photocurrent over plane",
        axis_tooltips=("x", "y", "current"),
        axis_units=("m", "m", "A"),
        db=False
    )

    plot_utils.plot_3d_and_top_views(
        x_matrix=x_matrix,
        y_matrix=y_matrix,
        z_matrix=snr,
        labels=(x_label, y_label, "SNR (dB)"),
        title="Expected SNR over plane",
        axis_tooltips=("x", "y", "SNR"),
        axis_units=("m", "m", "dB"),
        db=False
    )

    plot_utils.plot_3d_and_top_views(
        x_matrix=x_matrix,
        y_matrix=y_matrix,
        z_matrix=ber,
        labels=(x_label, y_label, "BER"),
        title="Expected BER over plane",
        axis_tooltips=("x", "y", "BER"),
        axis_units=("m", "m", ""),
        db=False
    )


if __name__ == '__main__':
    main()
