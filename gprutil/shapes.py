import numpy as np
from numbers import Number
from .materials import Material


class Point:
    def __init__(self, x, y, z):
        self.x = x
        self.y = y
        self.z = z

    def __str__(self):
        return f"{self.x} {self.y} {self.z}"


class Shape:
    def __init__(self, material: Material, smooth: bool = False):
        self.material = material
        self.smooth = smooth

    def __str__(self):
        return f"{self.material.identifier} {'y' if self.smooth else 'n'}"


class TriangularPrism(Shape):
    """Adds a triangular prism to the geometry."""

    def __init__(self, first_apex: Point, second_apex: Point, third_apex: Point, thickness, material: Material,
                 smooth: bool = False):
        super().__init__(material, smooth)
        self.first_apex = first_apex
        self.second_apex = second_apex
        self.third_apex = third_apex
        self.thickness = thickness

    def __str__(self):
        return f"#triangle: {self.first_apex} {self.second_apex} {self.third_apex} {self.thickness} {super().__str__()}"


class Triangle(TriangularPrism):
    """Adds a triangle to the geometry."""

    def __init__(self, first_apex: Point, second_apex: Point, third_apex: Point, material: Material,
                 smooth: bool = False):
        super().__init__(first_apex, second_apex, third_apex, 0, material, smooth)


class Box(Shape):
    """Adds an orthogonal parallelpiped to the geometry."""

    def __init__(self, lower_left: Point, upper_right: Point, material: Material, smooth: bool = False):
        super().__init__(material, smooth)
        self.lower_left = lower_left
        self.upper_right = upper_right

    def __str__(self):
        return f"#box {self.lower_left} {self.upper_right} {super().__str__()}"

    @property
    def height(self):
        """Height of the box in the z direction."""
        return self.upper_right.y - self.lower_left.y


class Sphere(Shape):
    """
    Adds a spherical object to the geometry. Sphere objects are permitted to extend outwith the model domain if
    desired, however, only parts of object inside the domain will be created.
    """

    def __init__(self, center: Point, radius: Number, material: Material, smooth: bool = False):
        super().__init__(material, smooth)
        self.center = center
        self.radius = radius

    def __str__(self):
        return f"#sphere {self.center} {self.radius} {super().__str__()}"


class Cylinder(Shape):
    """
    Adds a cylinder to the geometry. Cylinder objects are permitted to extend outwith the model domain if desired,
    however, only parts of object inside the domain will be created.
    """

    def __init__(self, face_center_1: Point, face_center_2: Point, radius: Number, material: Material,
                 smooth: bool = False):
        super().__init__(material, smooth)
        self.face_center_1 = face_center_1
        self.face_center_2 = face_center_2
        self.radius = radius

    def __str__(self):
        return f"{self.material}\n#cylinder: {self.face_center_1} {self.face_center_2} {self.radius}" \
               f" {super().__str__()}"


class CylindricalSector(Shape):
    pass

    # #cylindrical_sector:
    # Allows you to introduce a cylindrical sector (shaped like a slice of pie) into the model. The syntax of the command is:
    #
    # #cylindrical_sector: c1 f1 f2 f3 f4 f5 f6 f7 str1 [c1]
    # c1 is the direction of the axis of the cylinder from which the sector is defined and can be x, y, or z.
    # f1 f2 are the coordinates of the centre of the cylindrical sector.
    # f3 f4 are the lower and higher coordinates of the axis of the cylinder from which the sector is defined (in effect they specify the thickness of the sector).
    # f5 is the radius of the cylindrical sector.
    # f6 is the starting angle (in degrees) for the cylindrical sector (with zero degrees defined on the positive first axis of the plane of the cylindrical sector).
    # f7 is the angle (in degrees) swept by the cylindrical sector (the finishing angle of the sector is always anti-clockwise from the starting angle).
    # str1 is a material identifier that must correspond to material that has already been defined in the input file, or is one of the builtin materials pec or free_space.
    # c1 is an optional parameter which can be y or n, used to switch on and off dielectric smoothing.
    # For example, to specify a cylindrical sector with its axis in the z direction, radius of 0.25 m, thickness of 2 mm, a starting angle of 330 ∘, a sector angle of 60 ∘, and that is a perfect electric conductor, use: #cylindrical_sector: z 0.34 0.24 0.500 0.502 0.25 330 60 pec.
    #
    # Note
    #
    # Cylindrical sector objects are permitted to extend outwith the model domain if desired, however, only parts of object inside the domain will be created.


class FractalBox(Box):
    def __init__(self, lower_left: Point, upper_right: Point, material: Material, identifier: str,
                 fractal_dimension: Number = 1.5, weight_x: Number = 1, weight_y: Number = 1, weight_z: Number = 1,
                 surface_roughness: Number = 0, surface_water_depth: Number = 0):
        """
        1 f2 f3 are the lower left (x,y,z) coordinates of the parallelepiped, and f4 f5 f6 are the upper right (x,y,z) coordinates of the parallelepiped.
        f7 is the fractal dimension which, for an orthogonal parallelepiped, should take values between zero and three.
        f8 is used to weight the fractal in the x direction.
        f9 is used to weight the fractal in the y direction.
        f10 is used to weight the fractal in the z direction.
        i1 is the number of materials to use for the fractal distribution (defined according to the associated mixing model). This should be set to one if using a normal material instead of a mixing model.
        str1 is an identifier for the associated mixing model or material.
        str2 is an identifier for the fractal box itself.
        i2 is an optional parameter which controls the seeding of the random number generator used to create the fractals. By default (if you don’t specify this parameter) the random number generator will be seeded by trying to read data from /dev/urandom (or the Windows analogue) if available or from the clock otherwise.
        c1 is an optional parameter which can be y or n, used to switch on and off dielectric smoothing. If c1 is specified then a value for i2 must also be present.
        For example, to create an orthogonal parallelepiped with fractal distributed properties using a Peplinski mixing model for soil, with 50 different materials over a range of water volumetric fractions from 0.001 - 0.25, you should first define the mixing model using: #soil_peplinski: 0.5 0.5 2.0 2.66 0.001 0.25 my_soil and then specify the fractal box using #fractal_box: 0 0 0 0.1 0.1 0.1 1.5 1 1 1 50 my_soil my_fractal_box.

        Allows you to add rough surfaces to a #fractal_box in the model. A fractal distribution is used for the profile of the rough surface. The syntax of the command is:

        #add_surface_roughness: f1 f2 f3 f4 f5 f6 f7 f8 f9 f10 f11 str1 [i1]
        f1 f2 f3 are the lower left (x,y,z) coordinates of a surface on a #fractal_box, and f4 f5 f6 are the upper right (x,y,z) coordinates of a surface on a #fractal_box. The coordinates must locate one of the six surfaces of a #fractal_box but do not have to extend over the entire surface.
        f7 is the fractal dimension which, for an orthogonal parallelepiped, should take values between zero and three.
        f8 is used to weight the fractal in the first direction of the surface.
        f9 is used to weight the fractal in the second direction of the surface.
        f10 f11 define lower and upper limits for a range over which the roughness can vary. These limits should be specified relative to the dimensions of the #fractal_box that the rough surface is being applied.
        str1 is an identifier for the #fractal_box that the rough surface should be applied to.
        i1 is an optional parameter which controls the seeding of the random number generator used to create the fractals. By default (if you don’t specify this parameter) the random number generator will be seeded by trying to read data from /dev/urandom (or the Windows analogue) if available or from the clock otherwise.
        Up to six #add_rough_surface commands can be given for any #fractal_box corresponding to the six surfaces.

        #add_surface_water:
        Allows you to add surface water to a #fractal_box in the model that has had a rough surface applied. The syntax of the command is:

        #add_surface_water: f1 f2 f3 f4 f5 f6 f7 str1
        f1 f2 f3 are the lower left (x,y,z) coordinates of a surface on a #fractal_box, and f4 f5 f6 are the upper right (x,y,z) coordinates of a surface on a #fractal_box. The coordinates must locate one of the six surfaces of a #fractal_box but do not have to extend over the entire surface.
        f7 defines the depth of the water, which should be specified relative to the dimensions of the #fractal_box that the surface water is being applied.
        str1 is an identifier for the #fractal_box that the surface water should be applied to.
        For example, to add surface water that is 5 mm deep to an existing #fractal_box that has been specified using

        # #fractal_box: 0 0 0 0.1 0.1 0.1 1.5 1 1 1 50 my_soil my_fractal_box and has had a rough surface applied using
        # #add_surface_roughness: 0 0 0.1 0.1 0.1 0.1 1.5 1 1 0.085 0.110 my_fractal_box, use
        # #add_surface_water: 0 0 0.1 0.1 0.1 0.1 0.105 my_fractal_box.

        The water is modelled using a single-pole Debye formulation with properties ϵrs=80.1, ϵ∞=4.9, and a relaxation
        time of τ=9.231×10−12 seconds (http://dx.doi.org/10.1109/TGRS.2006.873208). If you prefer, gprMax will use your
        own definition for water as long as it is named water.

        :param lower_left:
        :param upper_right:
        :param material:
        :param identifier:
        :param fractal_dimension:
        :param weight_x:
        :param weight_y:
        :param weight_z:
        :param surface_roughness: surface roughness in mm
        :param surface_water_depth: depth of surface water in mm
        """

        if surface_water_depth and not surface_roughness:
            raise ValueError("Surface roughness must be added in order to add surface water.")

        super().__init__(lower_left, upper_right, material)
        self.identifier = identifier
        self.fractal_dimension = fractal_dimension
        self.weight_x = weight_x
        self.weight_y = weight_y
        self.weight_z = weight_z

        # Divide surface roughness by 2 in order to have both peaks and valleys within the overall roughness
        self.surface_roughness = surface_roughness / 1000 / self.height / 2
        self.surface_water_depth = surface_water_depth / 1000 / self.height

    def __str__(self):
        # TODO number of materials based on whether material is mixing model

        command = f"{self.material}\n"

        command += f"#fractal_box: {self.lower_left} {self.upper_right} {self.fractal_dimension} {self.weight_x} " \
               f"{self.weight_y} {self.weight_z} {self.material.n_materials} {self.material.identifier} {self.identifier}"

        if self.surface_roughness:
            command += f"\n#add_surface_roughness: {self.upper_left} {self.upper_right} {self.fractal_dimension} " \
                       f"{self.weight_x} {self.weight_y} {self.surface_roughness} {self.surface_roughness} " \
                       f"{self.identifier}"

        if self.surface_water_depth:
            command += f"\n#add_surface_water: {self.upper_left} {self.upper_right} {self.surface_water_depth} " \
                       f"{self.identifier}"

        return command

    @property
    def upper_left(self) -> Point:
        return Point(self.lower_left.x, self.lower_left.y, self.upper_right.z)


# #add_grass:
# Allows you to add grass with roots to a #fractal_box in the model. The blades of grass are randomly distributed over the specified surface area and a fractal distribution is used to vary the height of the blades of grass and depth of the grass roots. The syntax of the command is:
#
# #add_grass: f1 f2 f3 f4 f5 f6 f7 f8 f9 i1 str1 [i2]
# f1 f2 f3 are the lower left (x,y,z) coordinates of a surface on a #fractal_box, and f4 f5 f6 are the upper right (x,y,z) coordinates of a surface on a #fractal_box. The coordinates must locate one of three surfaces (in the positive axis direction) of a #fractal_box but do not have to extend over the entire surface.
# f7 is the fractal dimension which, for an orthogonal parallelepiped, should take values between zero and three.
# f8 f9 define lower and upper limits for a range over which the height of the blades of grass can vary. These limits should be specified relative to the dimensions of the #fractal_box that the grass is being applied.
# i1 is the number of blades of grass that should be applied to the surface area.
# str1 is an identifier for the #fractal_box that the grass should be applied to.
# i2 is an optional parameter which controls the seeding of the random number generator used to create the fractals. By default (if you don’t specify this parameter) the random number generator will be seeded by trying to read data from /dev/urandom (or the Windows analogue) if available or from the clock otherwise.
# For example, to apply 100 blades of grass that vary in height between 100 and 150 mm to the entire surface in the positive z direction of a #fractal_box that had been specified using #fractal_box: 0 0 0 0.1 0.1 0.1 1.5 1 1 50 my_soil my_fractal_box, use #add_grass: 0 0 0.1 0.1 0.1 0.1 1.5 0.2 0.25 100 my_fractal_box.
#
# Note
#
# The grass is modelled using a single-pole Debye formulation with properties ϵrs=18.5087, ϵ∞=12.7174, and a relaxation time of τ=1.0793×10−11 seconds (http://dx.doi.org/10.1007/BF00902994). If you prefer, gprMax will use your own definition for grass if you use a material named grass. The geometry of the blades of grass are defined by the parametric equations: x=xc+sx(tbx)2, y=yc+sy(tby)2, and z=t, where sx and sy can be -1 or 1 which are randomly chosen, and where the constants bx and by are random numbers based on a Gaussian distribution.


class Ground(FractalBox):
    """Represents the ground."""

    def __init__(self, length: Number, depth: Number, thickness: Number, material: Material,
                 surface_roughness: Number = 0, surface_water_depth: Number = 0):
        """
        Instantiates a new Ground object.

        :param length: length of geometry in the x direction
        :param thickness: thickness of geometry in the y direction
        :param depth: depth of soil in the z direction
        :param material: subsurface material
        :param surface_roughness: surface roughness in mm
        :param surface_water_depth: surface water depth in mm
        """

        # TODO: optional identifier to support layers of soil

        super().__init__(Point(0, 0, 0), Point(length, depth, thickness), material, "ground",
                         surface_roughness=surface_roughness, surface_water_depth=surface_water_depth)
