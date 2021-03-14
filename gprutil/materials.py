from numbers import Number


class Material:
    """Represents a material that can be added to a geometry."""

    def __init__(self, relative_permittivity: float, conductivity: float, relative_permeability: float,
                 magnetic_loss: float, identifier: str):
        """
        Instantiates a new Material.

        :param relative_permittivity: relative permittivity of the material
        :param conductivity: conductivity of the material in Siemens/meter
        :param relative_permeability: relative permeability of the material
        :param magnetic_loss: magnetic loss of the material in Ohms/meter
        :param identifier: identifier for the material
        """

        self.relative_permittivity = relative_permittivity
        self.conductivity = conductivity
        self.relative_permeability = relative_permeability
        self.magnetic_loss = magnetic_loss
        self.identifier = identifier

    def __str__(self):
        return f"#material: {self.relative_permittivity} {self.conductivity} {self.relative_permeability} " \
               f"{self.magnetic_loss} {self.identifier}"


class PerfectElectricConductor(Material):

    def __init__(self):
        super().__init__(0, 0, 0, 0, "pec")

    def __str__(self):
        return ''


class Air(Material):

    def __init__(self):
        super().__init__(0, 0, 0, 0, "air")

    def __str__(self):
        return ''


class Water(Material):

    def __init__(self):
        super().__init__(80, 4.194e-6, 1, 100, "water_user")


class Soil(Material):
    """
    Defines a soil using a mixing model proposed by Peplinski (http://dx.doi.org/10.1109/36.387598), valid for
    frequencies in the range 0.3GHz to 1.3GHz. Creates soils with realistic dielectric and geometric properties.
    """

    def __init__(self, sand: float, clay: float, bulk_density: Number, sand_density: Number, identifier: str,
                 water_min: float = 0, water_max: float = 0):
        """
        Instantiates a new Soil object.

        :param sand: the sand fraction of the soil
        :param clay: the clay fraction of the soil
        :param bulk_density: the bulk density of the soil in g/cm^3
        :param sand_density: the bulk density of the sand particles in g/cm^3
        :param identifier: unique identifier for the soil
        :param water_min: minimum volumetric water fraction of the soil
        :param water_max: maximum volumetric water fraction of the soil
        """

        super().__init__(0, 0, 0, 0, identifier)

        if water_min < 0 or water_min > 1:
            raise ValueError("Minimum water fraction must be in the range [0, 1].")
        if water_max < 0 or water_max > 1:
            raise ValueError("Maximum water fraction must be in the range [0, 1].")
        if sand + clay != 1:
            raise ValueError("Sum of the sand and clay fractions must be 1.")

        self.sand = sand
        self.clay = clay
        self.bulk_density = bulk_density
        self.sand_density = sand_density
        self.water_min = water_min
        self.water_max = water_max

    def __str__(self):
        return f"#soil_peplinski: {self.sand} {self.clay} {self.bulk_density} {self.sand_density} {self.water_min} " \
               f"{self.water_max} {self.identifier}"


class PVC(Material):
    def __init__(self):

        # https://passive-components.eu/what-is-dielectric-constant-of-plastic-materials/
        # http://docs.gprmax.com/en/latest/input.html
        # https://www.sciencedirect.com/science/article/pii/S0167273800005981

        super().__init__(4, 10e-6, 1, 0, 'pvc')


# class CarbonSteel(Material):
#
#     def __init__(self, relative_permittivity: object, conductivity: object, relative_permeability: object,
#                  magnetic_loss: object, identifier: object):
#
#         # https: // www.engineeringtoolbox.com / permeability - d_1923.html
#         super().__init__(0, 1.3e6, 100, magnetic_loss, identifier)
