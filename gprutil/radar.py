from gprutil.shapes import Point


class Waveform:
    """
    Defines a waveform that can be transmitted by a radar transmitter/=.

    Available waveform types are:
        gaussian: Gaussian waveform.
        gaussiandot: first derivative of a Gaussian waveform.
        gaussiandotnorm: normalised first derivative of a Gaussian waveform.
        gaussiandotdot: second derivative of a Gaussian waveform.
        gaussiandotdotnorm: normalised second derivative of a Gaussian waveform.
        ricker: Ricker waveform, i.e. the negative, normalised second derivative of a Gaussian waveform.
        gaussianprime: first derivative of a Gaussian waveform, directly derived from the aforementioned gaussian
        gaussiandoubleprime: second derivative of a Gaussian waveform, directly derived from the aforementioned
            gaussian
        sine: a single cycle of a sine waveform.
        contsine: a continuous sine waveform. In order to avoid introducing noise into the calculation the amplitude
            of the waveform is modulated for the first cycle of the sine wave (ramp excitation).


#waveform: str1 f1 f2 str2
str1 is the type of waveform which can be:

f1 is the scaling of the maximum amplitude of the waveform
(for a #hertzian_dipole the units will be Amps, for a #voltage_source or #transmission_line the units will be Volts).

f2 is the centre frequency of the waveform (Hertz). In the case of the Gaussian waveform it is related to the pulse width.

str2 is an identifier for the waveform used to assign it to a source.
    """

    def __init__(self, waveform_type, amplitude_scaling, center_frequency, identifier):
        """
        Instantiates a new Waveform.

        :param waveform_type: the type of waveform
        :param amplitude_scaling: scaling of the maximum amplitude of the waveform  (for a hertzian dipole the units
                                  will be Amps, for a voltage source or transmission line the units will be Volts).
        :param center_frequency: center frequency of the waveform (Hertz). In the case of the Gaussian waveform it is
                                 related to the pulse width.
        :param identifier: a unique identifier for the waveform
        :raises ValueError: if an invalid waveform type is provided
        """

        if waveform_type not in ['gaussian', 'gaussiandot', 'gaussiandotnorm', 'gaussiandotdot', 'gaussiandotdotnorm',
                                 'ricker', 'gaussianprime', 'gaussiandoubleprime', 'sine', 'contsine']:
            raise ValueError("Invalid waveform type.")

        self.waveform_type = waveform_type
        self.amplitude_scaling = amplitude_scaling
        self.center_frequency = center_frequency
        self.identifier = identifier

    def __str__(self):
        return f"#waveform: {self.waveform_type} {self.amplitude_scaling} {self.center_frequency:e} {self.identifier}"


class RickerWaveform(Waveform):
    """Represents a Ricker waveform."""

    def __init__(self, amplitude_scaling, center_frequency, identifier):
        """
        Instantiates a new RickerWaveform.

        :param amplitude_scaling: scaling of the maximum amplitude of the waveform  (for a hertzian dipole the units
                                  will be Amps, for a voltage source or transmission line the units will be Volts).
        :param center_frequency: center frequency of the waveform (Hertz) - related to the pulse width.
        :param identifier: a unique identifier for the waveform
        """
        super().__init__('ricker', amplitude_scaling, center_frequency, identifier)


class Transmitter:
    """Represents a radar transmitter antenna."""

    def __init__(self, polarization: str, location: Point, waveform: Waveform):
        """
        Instantiates a new Transmitter.

        :param polarization: polarization of the transmitter - must be one of 'x', 'y', or 'z'
        :param location: location of the transmitter
        :param waveform: the waveform to be transmitted by this transmitter
        :raises ValueError: if an invalid polariation is provided
        """

        if polarization not in ['x', 'y', 'z']:
            raise ValueError("Invalid polarization - must be one of 'x', 'y', or 'z'.")

        self.polarization = polarization
        self.location = location
        self.waveform = waveform

    def __str__(self):
        return f"{self.polarization} {self.location} {self.waveform.identifier}"


class HertzianDipoleTransmitter(Transmitter):
    """
    Represents a Hertzian dipole radar transmitter. Allows specification of a current density term at an
    electric field location - the simplest excitation, often referred to as an additive or soft source.
    """

    def __str__(self):
        return f"#hertzian_dipole: {super().__str__()}"


class Receiver:
    """Represents a receiver antenna."""

    def __init__(self, location: Point, identifier: str):
        """
        Instantiates a new Receiver.

        :param location: the location of the receiver
        :param identifier: a unique identifier for the receiver
        """

        # TODO: error if location is outside of scene?

        self.location = location
        self.identifier = identifier

    def __str__(self):
        return f"#rx: {self.location}"
