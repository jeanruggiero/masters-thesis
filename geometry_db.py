import os
from skopt.space import Space
from skopt.sampler import Lhs
import pandas as pd
import numpy as np

import psycopg2
from psycopg2 import OperationalError


def create_connection(db_name, db_user, db_password, db_host, db_port):
    connection = None
    try:
        connection = psycopg2.connect(
            database=db_name,
            user=db_user,
            password=db_password,
            host=db_host,
            port=db_port,
        )
        print("Connection to PostgreSQL DB successful")
    except OperationalError as e:
        print(f"The error '{e}' occurred")
    return connection

def execute_query(connection, query):
    connection.autocommit = True
    cursor = connection.cursor()
    try:
        cursor.execute(query)
        print("Query executed successfully")
    except OperationalError as e:
        print(f"The error '{e}' occurred")

# connection = create_connection('postgres', 'jean', os.environ.get('DB_PASSWORD'),
#                                'masters-thesis.cpna5xaoi8xu.us-west-2.rds.amazonaws.com', 5432)

# create_table = """
# CREATE TABLE IF NOT EXISTS geometries (
#   id SERIAL PRIMARY KEY,
#   radius FLOAT NOT NULL,
#   depth FLOAT NOT NULL,
#   rx_tx_height FLOAT NOT NULL,
#   cylinder_material TEXT NOT NULL,
#   cylinder_fill_material TEXT NOT NULL
# )
# """
#
# execute_query(connection, create_table)





# Number of simulations to run
n_sim = 1000

width = 3 # domain x
depth = 3.25 # domain y
thickness = 0.002 # domain z

air_space = 0.25  # Amount of air above ground in the domain

space = Space([
    # (0.005, 0.25),  # radius
    (0.1, 0.9),  # sand proportion
    (1.5, 3),  # soil bulk density
    (1, 5),  # sand particle bulk density
    # (1.0, float(depth - air_space - 1)),  # cylinder depth
    (0.01, 0.2),  # rx/tx height above ground
    # (0.005, 0.5),  # step size
    (0.0, 0.1),  # surface roughness
    (0, 1),  # surface water depth
    # [0, 1],  # categorical cylinder material
    # [0, 1],  # categorical cylinder fill material
    (2e8, 2.7e8, 3.5e8, 4e8, 5e8, 6e8, 8e8, 9e8)
])

lhs = Lhs(lhs_type="classic", criterion=None)
labels = lhs.generate(space.dimensions, n_sim)

records = []
for (sand_proportion, soil_density, sand_particle_density, rx_tx_height, surface_roughness, surface_water_depth,
     frequency) in labels:
    records.append((sand_proportion, soil_density, sand_particle_density, rx_tx_height, surface_roughness, surface_water_depth,
     frequency))

df = pd.DataFrame(labels, columns=['sand_proportion', 'soil_density', 'sand_particle_density', 'rx_tx_height',
                                   'surface_roughness', 'surface_water_depth', 'frequency'],
                  index=np.arange(20000, 20000 + len(labels)))

df.to_csv('geometry_spec_negative.csv')
# for i in range(0)
#
# df[:500].to_csv('geometry_spec1.csv')
# df[500:].to_csv('geometry_spec2.csv')
#
#
# records = ", ".join(["%s"] * len(records))
# insert_query = (
#     f"INSERT INTO geometries (radius, depth, rx_tx_height, cylinder_material, cylinder_fill_material) VALUES {records}"
# )
# connection.autocommit = True
# cursor = connection.cursor()
# cursor.execute(insert_query, records)
# #
# # geometries = []
# #
# connection.close()
#
# # for i, (radius, cylinder_depth, rx_tx_height, cylinder_material_type, fill_material_type) in enumerate(labels):