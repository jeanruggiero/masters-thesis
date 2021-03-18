# This module contains code that generates gprMax geometry files given a set of parameters.
import numpy as np
import math
import os
import textwrap

# import gprMax.input_cmd_funcs as gprmax

from typing import Sequence, Type
from numbers import Number

from .shapes import Shape, Ground, Point
from .radar import Waveform, Receiver, HertzianDipoleTransmitter


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

    # @property
    # def dt_max(self) -> float:
    #     """The maximum permissible time step for this Discretization."""
    #     return 1 / (gprmax.c * np.sqrt(1/self.dx**2 + 1/self.dy**2 + 1/self.dz**2))

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
        return f"#time_window: {self.time_window:e}"


class Geometry:

    def __init__(self, title: str, geometry_path: str, scan_path: str, domain: Domain, discretization: Discretization,
                 time_window: TimeWindow, ground: Ground, shapes: Sequence[Shape], waveform: Waveform,
                 transmitter_location: Point = None, receiver_location: Point = None, step_size: Number = 0.002,
                 geometry_view: bool = False, cores=2):

        self.title = title
        self.geometry_path = geometry_path
        self.scan_path = scan_path
        self.domain = domain
        self.discretization = discretization
        self.time_window = time_window
        self.ground = ground
        self.shapes = shapes
        self.waveform = waveform
        self.geometry_view = geometry_view
        self.cores = cores

        # Both relative to lower left corner of free space above ground
        if transmitter_location:
            self.transmitter = HertzianDipoleTransmitter(
                'z', Point(transmitter_location.x, transmitter_location.y + ground.height,
                           transmitter_location.z),
                self.waveform
            )
        else:
            self.transmitter = HertzianDipoleTransmitter(
                'z', Point(0.04, 0.02 + ground.height, 0), self.waveform
            )

        if receiver_location:
            self.receiver = Receiver(
                Point(receiver_location.x, receiver_location.y + ground.height, receiver_location.z), 'receiver'
            )
        else:
            self.receiver = Receiver(Point(0.08, 0.02 + ground.height, self.domain.size_z / 2), 'receiver')

        self.step_size = step_size

    def generate(self):
        with open(os.path.join(self.geometry_path, self.title + ".in"), 'w') as f:
            f.write(textwrap.dedent(
                f"""\
                #title: {self.title}
                {self.domain}
                {self.discretization}
                {self.time_window}
                
                #messages: n
                #output_dir: ../{self.scan_path}
                
                {self.waveform}
                {self.transmitter}
                {self.receiver}                
                """
            ))

            if self.step_size:
                f.write(textwrap.dedent(
                    f"""\
                    #src_steps: {self.step_size} 0 0
                    #rx_steps: {self.step_size} 0 0
                    """
                ))

            f.write(str(self.ground) + '\n')

            f.write("\n".join(str(shape) for shape in self.shapes))

            if self.geometry_view:
                f.write(f"\n#geometry_view: 0 0 0 {self.domain.size_x} {self.domain.size_y} {self.domain.size_z} " +
                        f"{self.discretization.dx} {self.discretization.dy} {self.discretization.dz} " +
                        f"{self.title}_view n")

            if self.cores:
                f.write(f'\n#num_threads: {self.cores}')

    @property
    def steps(self):
        return math.floor((self.domain.size_x - self.transmitter.location.x - self.discretization.dx * 10 - 0.04) /
                          self.step_size)
