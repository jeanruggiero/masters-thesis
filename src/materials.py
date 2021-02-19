class Material:
    """Represents a material that can be added to a geometry."""

    def __init__(self, relative_permittivity: Number, conductivity: Number, relative_permeability: Number,
                 magnetic_loss: Number, identifier: str):
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
        return f"material: {self.relative_permittivity} {self.conductivity} {self.relative_permeability} " \
               f"{self.magnetic_loss} {self.identifier}"


class PerfectElectricConductor(Material):
    pass

class Air(Material):
    pass

class MixingModel(Material):
    pass

class Soil(MixingModel):
    pass