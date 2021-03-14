from gprutil.geometry import Geometry, Domain, Discretization, TimeWindow
from gprutil.shapes import Ground, Cylinder, Point
from gprutil.materials import PerfectElectricConductor, Soil, Material, PVC, Water, Air
from gprutil.radar import Transmitter, Receiver, RickerWaveform

import os

from skopt.space import Space
from skopt.sampler import Lhs

n_sim = 12

width = 3 # domain x
depth = 3.25 # domain y
thickness = 0.002 # domain z

air_space = 0.25  # Amount of air above ground in the domain

# Cylinder wall thickness as a % of radius
wall_thickness = 0.15

geometry_path = 'geometry'
scan_path = '../simulations'

space = Space([
    (0.005, 0.25),  # radius
    # (0.1, 0.9),  # sand proportion
    # (1.5, 3),  # soil bulk density
    # (1, 5),  # sand particle bulk density
    (1, float(depth - air_space - 1)),  # cylinder depth
    (0.01, 0.2),  # rx/tx height above ground
    # (0.005, 0.5),  # step size
    # (0.0, 0.1),  # surface roughness
    # (0, 1),  # surface water depth
    [0, 1],  # categorical cylinder material
    [0, 1],  # categorical cylinder fill material
    (2e8, 2.7e8, 3.5e8, 4e8, 5e8, 6e8, 8e8, 9e8)
])

lhs = Lhs(lhs_type="classic", criterion=None)
labels = lhs.generate(space.dimensions, n_sim)

geometries = []

for i, (radius, cylinder_depth, rx_tx_height, cylinder_material_type, fill_material_type) in enumerate(labels):

    cylinder_x = width / 2
    surface_roughness = 0.02
    # rx_tx_height = 0
    #ground_material = Soil(0.5, 0.5, 2, 2.66, "balanced_soil", 0.001, 0.25)
    ground_material = Material(6, 0, 1, 0, 'half_space')
    cylinder_z = depth - air_space - cylinder_depth - radius

    # Carbon Steel
    # https: // www.engineeringtoolbox.com / permeability - d_1923.html
    cylinder_material = PerfectElectricConductor() if cylinder_material_type else PVC()
    fill_material = Air() if fill_material_type else Water()

    cylinder = Cylinder(Point(cylinder_x, cylinder_z, 0), Point(cylinder_x, cylinder_z, 0.002), radius,
                        cylinder_material)
    cylinder_fill = Cylinder(Point(cylinder_x, cylinder_z, 0), Point(cylinder_x, cylinder_z, 0.002),
                             radius * (1 - wall_thickness), fill_material)
    spatial_resolution = 0.002

    geo = Geometry(
        f'test_cylinder_{i}',
        os.path.join(geometry_path, 'test'),
        os.path.join(scan_path, 'test'),
        Domain(width, depth, thickness),
        Discretization(spatial_resolution, spatial_resolution, spatial_resolution),
        TimeWindow(1.2e-7),
        Ground(width, depth - air_space, thickness, ground_material, surface_roughness=surface_roughness),
        [cylinder, cylinder_fill],
        RickerWaveform(1, 2.5e8, 'ricker'),
        step_size=0.02,
        receiver_location=Point(spatial_resolution * 10, rx_tx_height, 0),
        transmitter_location=Point(spatial_resolution * 10 + 0.04, rx_tx_height, 0)
    )

    geo.generate()
    geometries.append(geo)

print([geo.steps for geo in geometries])