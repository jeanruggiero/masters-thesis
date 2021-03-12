from gprutil.geometry import Geometry, Domain, Discretization, TimeWindow
from gprutil.shapes import Ground, Cylinder, Point
from gprutil.materials import PerfectElectricConductor, Soil, Material
from gprutil.radar import Transmitter, Receiver, RickerWaveform

import os

from skopt.space import Space
from skopt.sampler import Lhs

n_sim = 12

width = 1 # domain x
depth = 3 # domain y
thickness = 0.002 # domain z

air_space = 1  # Amount of air above ground in the domain

geometry_path = 'geometry'
scan_path = '../simulations'

space = Space([
    (0.01, 1.0),  # radius
    # (0.1, 0.9),  # sand proportion
    # (1.5, 3),  # soil bulk density
    # (1, 5),  # sand particle bulk density
    (0.01, float(depth - air_space - 1)),  # cylinder depth
    (0.0, float(width)),  # cylinder x position
    # (0.5, 2.5),  # rx/tx z position
    (0.01, 0.8),  # rx/tx height above ground
    # (0.005, 0.5),  # step size
    (0.0, 0.1),  # surface roughness
    # (0, 1),  # surface water depth
])

lhs = Lhs(lhs_type="classic", criterion=None)
labels = lhs.generate(space.dimensions, n_sim)

geometries = []

for i, (radius, cylinder_depth, cylinder_x, rx_tx_height, surface_roughness) in enumerate(labels):

    rx_tx_height = 0
    #ground_material = Soil(0.5, 0.5, 2, 2.66, "balanced_soil", 0.001, 0.25)
    ground_material = Material(6, 0, 1, 0, 'half_space')
    cylinder_z = depth - air_space - cylinder_depth - radius
    cylinder = Cylinder(Point(cylinder_x, cylinder_z, 0), Point(cylinder_x, cylinder_z, 0.002), radius,
                        PerfectElectricConductor())
    spatial_resolution = 0.002

    geo = Geometry(
        f'test_cylinder_{i}',
        os.path.join(geometry_path, 'test'),
        os.path.join(scan_path, 'test'),
        Domain(width, depth, thickness),
        Discretization(spatial_resolution, spatial_resolution, spatial_resolution),
        TimeWindow(3e-8),
        Ground(width, depth - air_space, thickness, ground_material, surface_roughness=surface_roughness),
        [cylinder],
        RickerWaveform(1, 9e8, 'ricker'),
        step_size=0.02,
        receiver_location=Point(spatial_resolution * 10, rx_tx_height, 0),
        transmitter_location=Point(spatial_resolution * 10 + 0.04, rx_tx_height, 0)
    )

    geo.generate()
    geometries.append(geo)

print([geo.steps for geo in geometries])