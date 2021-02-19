# This module contains code that generates gprMax geometry files given a set of parameters.
import numpy as np
import gprMax.input_cmd_funcs as gprmax

from typing import Sequence
from numbers import Number

from shapes import Shape, Ground
from radar import Waveform, Transmitter, Receiver


class Domain:
    """Specifies the entire domain of the geometry."""

    def __init__(self, size_x: Number, size_y: Number, size_z: Number):
        """
        Instantiates a new Domain.

        :param size_x: size of the domain in the x dimension in meters
        :param size_y: size of the domain in the y dimension in meters
        :param size_z: size of the domain in the z dimension in meters
        """
        self.size_x = size_x
        self.size_y = size_y
        self.size_z = size_z

    def __str__(self) -> str:
        return f"#domain: {self.size_x} {self.size_y} {self.size_z}"


class Discretization:
    """Represents the discretization of a geometry."""

    def __init__(self, dx: Number, dy: Number, dz: Number):
        """
        Instantiates a new Discretization.

        :param dx: spatial step in the x direction in meters
        :param dy: spatial step in the y direction in meters
        :param dz: spatial step in the z direction in meters
        """

        self.dx = dx
        self.dy = dy
        self.dz = dz

    @property
    def dt_max(self) -> float:
        """The maximum permissible time step for this Discretization."""
        return 1 / (gprmax.c * np.sqrt(1/self.dx**2 + 1/self.dy**2 + 1/self.dz**2))

    def __str__(self) -> str:
        return f"#dx_dy_dz: {self.dx} {self.dy} {self.dz}"


class TimeWindow:
    """The total required simulated time."""

    def __init__(self, time_window: Number):
        """
        Instantiates a new TimeWindow.

        :param time_window: the total required simulated time in seconds
        """
        self.time_window = time_window

    def __str__(self) -> str:
        return f"#time_window: {self.time_window:f}"


class Geometry:

    def __init__(self, title: str, geometry_path: str, scan_path: str, domain: Domain, discretization: Discretization,
                 time_window: TimeWindow, ground: Ground, shapes: Sequence[Shape], waveform: Waveform,
                 transmitter: Transmitter, receiver: Receiver):

        self.title = title
        self.geometry_path = geometry_path
        self.scan_path = scan_path
        self.domain = domain
        self.discretization = discretization
        self.time_window = time_window
        self.ground = ground
        self.shapes = shapes
        self.waveform = waveform
        self.transmitter = transmitter
        self.receiver = receiver

    def generate(self):
        with open(self.geometry_path / self.title + ".in") as f:
            f.write(
                f"""
                {self.title}
                {self.domain}
                {self.discretization}
                {self.time_window}
                
                #messages: n
                #output_dir: ../{self.scan_path}
                #num_threads: 3
                
                {self.waveform}
                {self.transmitter}
                {self.receiver}
                
                {self.ground}
                """
            )

            f.write("\n".join(str(shape) for shape in self.shapes))
