from gprutil.geometry import Geometry, Domain, Discretization, TimeWindow
from gprutil.shapes import Ground, Cylinder, Point
from gprutil.materials import PerfectElectricConductor, Soil, Material, PVC, Water, Air
from gprutil.radar import Transmitter, Receiver, RickerWaveform

import os
import glob
import subprocess
import h5py

import boto3
import botocore

import pandas as pd


def make_scene(id, ascan_number, step_size, radius, cylinder_depth, rx_tx_height, cylinder_material_type,
               fill_material_type, geometry_path, scan_path, frequency=2.5e8):
    width = 3  # domain x
    depth = 3.25  # domain y
    thickness = 0.002  # domain z
    air_space = 0.25  # Amount of air above ground in the domain

    # Cylinder wall thickness as a % of radius
    wall_thickness = 0.15

    cylinder_x = width / 2
    surface_roughness = 0.02
    # ground_material = Soil(0.5, 0.5, 2, 2.66, "balanced_soil", 0.001, 0.25)
    ground_material = Material(6, 0, 1, 0, 'half_space')
    cylinder_z = depth - air_space - cylinder_depth - radius

    # Carbon Steel
    # https: // www.engineeringtoolbox.com / permeability - d_1923.html
    cylinder_material = PerfectElectricConductor() if cylinder_material_type == 'pec' else PVC()
    fill_material = Air() if fill_material_type == 'air' else Water()

    cylinder = Cylinder(Point(cylinder_x, cylinder_z, 0), Point(cylinder_x, cylinder_z, 0.002), radius,
                        cylinder_material)
    cylinder_fill = Cylinder(Point(cylinder_x, cylinder_z, 0), Point(cylinder_x, cylinder_z, 0.002),
                             radius * (1 - wall_thickness), fill_material)
    spatial_resolution = 0.002

    geo = Geometry(
        f'test_cylinder_{id}',
        geometry_path,
        scan_path,
        Domain(width, depth, thickness),
        Discretization(spatial_resolution, spatial_resolution, spatial_resolution),
        TimeWindow(1.2e-7),
        Ground(width, depth - air_space, thickness, ground_material, surface_roughness=surface_roughness),
        [cylinder, cylinder_fill],
        RickerWaveform(1, frequency, 'ricker'),
        step_size=0,
        receiver_location=Point(spatial_resolution * 10 + step_size * ascan_number, rx_tx_height, 0),
        transmitter_location=Point(spatial_resolution * 10 + 0.04 + step_size * ascan_number, rx_tx_height, 0)
    )

    geo.generate()


def scan_exists(id, ascan_number):

    s3 = boto3.resource('s3')

    try:
        s3.Object('jean-masters-thesis', f'simulations/{id}/scan{ascan_number}.out').load()
    except botocore.exceptions.ClientError as e:
        if e.response['Error']['Code'] == "404":
            return False
    return True


if __name__ == '__main__':

    geometries = pd.read_csv('geometry_spec.csv', index_col=0)

    geometry_path = 'geometry'
    scan_path = 'simulations'

    for id, geometry in geometries.iterrows():
        for ascan_number in range(0, 144):

            # Check if scan exists first
            if scan_exists(id, ascan_number):
                continue

            input_filename = geometry_path + f'/test_cylinder_{id}.in'

            # Generate input file
            try:
                make_scene(
                    id, ascan_number, 0.02, geometry['radius'], geometry['depth'], geometry['rx_tx_height'],
                    geometry['cylinder_material'], geometry['cylinder_fill_material'], geometry_path, scan_path,
                    geometry['frequency']
                )
            except KeyError:
                make_scene(
                    id, ascan_number, 0.02, geometry['radius'], geometry['depth'], geometry['rx_tx_height'],
                    geometry['cylinder_material'], geometry['cylinder_fill_material'], geometry_path, scan_path, 2.5e8
                )

            # Run simulation
            os.system(f'python3 -m gprMax {input_filename}')

            # Delete input file
            os.remove(input_filename)

            # Copy output file to s3
            os.system(f'aws s3 cp simulations/test_cylinder_{id}.out s3://jean-masters-thesis/simulations/{id}/scan'
                      f'{ascan_number}.out')

            os.remove(f'simulations/test_cylinder_{id}.out')
