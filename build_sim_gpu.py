from build_sim import make_scene, scan_exists
import pandas as pd
import os


if __name__ == '__main__':

    geometries = pd.read_csv('geometry_spec.csv', index_col=0)
    n_cores = 8

    geometry_path = 'geometry'
    scan_path = 'simulations'

    for id, geometry in geometries.iterrows():
        if id < 1100:
            continue

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
                    geometry['frequency'], cores=n_cores
                )
            except KeyError:
                make_scene(
                    id, ascan_number, 0.02, geometry['radius'], geometry['depth'], geometry['rx_tx_height'],
                    geometry['cylinder_material'], geometry['cylinder_fill_material'], geometry_path, scan_path,
                    2.5e8, cores=n_cores
                )

            # Run simulation
            os.system(f'python3 -m gprMax {input_filename} -gpu')

            # Delete input file
            os.remove(input_filename)

            # Copy output file to s3
            os.system(f'aws s3 cp simulations/test_cylinder_{id}.out s3://jean-masters-thesis/simulations/{id}/scan'
                      f'{ascan_number}.out')

            os.remove(f'simulations/test_cylinder_{id}.out')
