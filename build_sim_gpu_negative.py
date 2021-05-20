from build_sim import make_scene, make_scene_negative, scan_exists
import pandas as pd
import os
import multiprocessing


def run_sim(args):

    id = args[0]
    ascan_number = args[1]
    geometry = args[2]

    geometry_path = 'geometry'
    scan_path = 'simulations'

    n_cores = 8

    # Check if scan exists first
    if scan_exists(id, ascan_number):
        return

    print(f"Running scan {id}.{ascan_number}.")

    input_filename = geometry_path + f'/test_cylinder_{id}_{ascan_number}.in'

    # Generate input file
    try:
        make_scene(
            f'{id}_{ascan_number}', ascan_number, 0.02, geometry['radius'], geometry['depth'], geometry['rx_tx_height'],
            geometry['cylinder_material'], geometry['cylinder_fill_material'], geometry_path, scan_path,
            geometry['frequency'], cores=n_cores
        )
    except KeyError:
        make_scene(
            f'{id}_{ascan_number}', ascan_number, 0.02, geometry['radius'], geometry['depth'], geometry['rx_tx_height'],
            geometry['cylinder_material'], geometry['cylinder_fill_material'], geometry_path, scan_path,
            2.5e8, cores=n_cores
        )

    # Run simulation
    os.system(f'python3 -m gprMax {input_filename} -gpu')

    # Delete input file
    os.remove(input_filename)

    # Copy output file to s3
    os.system(f'aws s3 cp simulations/test_cylinder_{id}_{ascan_number}.out s3://jean-masters-thesis/simulations/{id}/scan'
              f'{ascan_number}.out --quiet')

    os.remove(f'simulations/test_cylinder_{id}_{ascan_number}.out')


def run_sim_negative(args):

    id = args[0]
    ascan_number = args[1]
    geometry = args[2]

    geometry_path = 'geometry'
    scan_path = 'simulations'

    n_cores = 8

    # Check if scan exists first
    if scan_exists(id, ascan_number):
        return

    print(f"Running scan {id}.{ascan_number}.")

    input_filename = geometry_path + f'/test_cylinder_{id}_{ascan_number}.in'

    # Generate input file
    try:
        make_scene_negative(
            f'{id}_{ascan_number}', ascan_number, 0.02, geometry['sand_proportion'], geometry['soil_density'], geometry['sand_particle_density'],
            geometry['rx_tx_height'], geometry['surface_roughness'], geometry['surface_water_depth'], geometry_path, scan_path,
            geometry['frequency'], cores=n_cores
        )
    except KeyError:
        make_scene_negative(
            f'{id}_{ascan_number}', ascan_number, 0.02, geometry['sand_proportion'], geometry['soil_density'],
            geometry['sand_particle_density'], geometry['rx_tx_height'], geometry['surface_roughness'],
            geometry['surface_water_depth'], geometry_path, scan_path, 2.5e8, cores=n_cores
        )

    # Run simulation
    os.system(f'python3 -m gprMax {input_filename} -gpu')

    # Delete input file
    os.remove(input_filename)

    # Copy output file to s3
    os.system(f'aws s3 cp simulations/test_cylinder_{id}_{ascan_number}.out s3://jean-masters-thesis/simulations/{id}/scan'
              f'{ascan_number}.out --quiet')

    os.remove(f'simulations/test_cylinder_{id}_{ascan_number}.out')


if __name__ == '__main__':

    geometries = pd.read_csv('geometry_spec.csv', index_col=0)

    for (id, geometry) in geometries.iterrows():
        args = ((id, asn, geometry) for asn in range(144))

        with multiprocessing.Pool(8) as p:
            p.map(run_sim_negative, args)

    # args = ((id, asn, geometry) for asn in range(144) for (id, geometry) in geometries.iterrows())
    #
    # with multiprocessing.Pool(8) as p:
    #     p.map(run_sim, args)
