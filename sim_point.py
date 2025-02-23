from enum import Enum, auto
from typing import Tuple
import math

class Element(Enum):
    HELIUM = auto() # helium-4
    OXYGEN = auto() # O_2
    IRON = auto()

    @property
    def molecular_mass(self) -> float:
        # In g/mol
        if self == Element.HELIUM:
            return 4.002603
        if  self == Element.OXYGEN:
            return 31.9988
        if self == Element.IRON:
            return 55.85
        assert False

    def absorption_cross_section(self, wavelength: float) -> float:
        # These values most hold for the visible spectrum since that's what
        # we're working with.
        if self == Element.HELIUM:
            return 10.0e-30
        if self == Element.OXYGEN:
            return 10.0e-22
        if self == Element.IRON:
            if abs(wavelength - 438.3) < 10.0 or abs(wavelength - 526.9) < 10.0 or abs(wavelength - 672.0) < 10.0:
                return 10.0e-13
            else:
                return 10.0e-16
        assert False

def scale_height(temp: float, mean_molecular_mass: float, gravity: float) -> float:
    # H = RT / Mg where R is the molar gas constant and M is the mean moral mass
    R = 8.31446
    return R * temp / (mean_molecular_mass * gravity)

def pressure(temp: float, height: float, mean_molecular_mass: float, gravity: float, surface_pressure: float) -> float:
    scale_height_ = scale_height(temp, mean_molecular_mass, gravity)
    return surface_pressure * math.exp(-height / scale_height_)

GRAVITY = 9.81

def elem_dist_by_height(height: float, temp: float, surface_pressure: float, elements: list[Element]) -> list[Tuple[Element, float]]:
    pressures = [pressure(temp, height, elem.molecular_mass, GRAVITY, surface_pressure) for elem in elements]
    total_pressure = sum(pressures)
    return [(elem, pressure / total_pressure if total_pressure != 0.0 else 0.0) for elem, pressure in zip(elements, pressures)]

def planck_func(wavelength: float, temp: float) -> float:
    h = 6.626e-34  # Planck constant (J s)
    c = 3e8  # Speed of light (m/s)
    k_B = 1.380649e-23  # Boltzmann constant (J/K)
    return 2 * h * c**2 / wavelength**5 / (math.exp(h * c / (wavelength * k_B * temp)) - 1)

def calc_spectra(dist_by_layer: list[list[Tuple[Element, float]]], wavelengths: list[float], layer_depth: float, temp: float) -> Tuple[list[list[float]], list[list[float]]]:
    emission_spectra = []
    transmitted_spectra = []

    for dist in dist_by_layer:
        emission_spectrum = []
        transmitted_spectrum = []
        for wavelength in wavelengths:
            absorption_coeff = sum(frac * elem.absorption_cross_section(wavelength) for elem, frac in dist)
            optical_depth = absorption_coeff * layer_depth # TODO: Change layer_depth here depending on angle of incoming ray. Assuming 90 degrees to surface now.
            transmission = math.exp(-optical_depth)
            absorption = 1 - transmission
            emission_spectrum.append(absorption * planck_func(wavelength, temp))
            transmitted_spectrum.append(transmission)

        emission_spectra.append(emission_spectrum)
        transmitted_spectra.append(transmitted_spectrum)

    return emission_spectra, transmitted_spectra

def sim_point(temp: float, surface_pressure: float, elements: list[Tuple[Element, float]], wavelengths: list[float], incident_spectrum: list[float], heights: list[float], depth: float) -> list[float]:
    assert sum(cont for _, cont in elements) == 1.0
    assert len(wavelengths) == len(incident_spectrum)

    dist_by_layer = [elem_dist_by_height(height, temp, surface_pressure, list(map(lambda x: x[0], elements))) for height in heights]

    emission_spectra, transmitted_spectra = calc_spectra(dist_by_layer, wavelengths, depth, temp)

    #  For each layer, the spectrum that's transmitted through all layers above
    transmitted_above = []
    for layer_idx, emission_spectrum in enumerate(emission_spectra):
        assert len(emission_spectrum) == len(wavelengths)
        transmitted_above.append([emitted * math.prod(map(lambda ts: ts[wavelength_idx], transmitted_spectra[layer_idx + 1:])) for wavelength_idx, emitted in enumerate(emission_spectrum)])

    # This is how much of the stars spectrum reaches the surface
    incident_surface_spectrum = incident_spectrum.copy()
    for transmission_spectrum in transmitted_spectra:
        for idx, transmitted in enumerate(transmission_spectrum):
            incident_surface_spectrum[idx] *= transmitted

    # spectrum = incident_surface_spectrum # The spectrum reflected from the surface acts as our base
    spectrum = [0.0 for _ in range(len(wavelengths))]
    for layer in transmitted_above:
        for idx, transmitted in enumerate(layer):
            spectrum[idx] += transmitted

    assert len(spectrum) == len(wavelengths)

    return spectrum


def calc_and_show():
    temp = 270.0
    surface_pressure = 1.0
    elements = [(Element.HELIUM, 0.7), (Element.OXYGEN, 0.1), (Element.IRON, 0.2)]

    wavelengths = [i + 400.0 for i in range(400)]
    incident_spectrum = [1.0 for _ in range(400)]
    depth = 10.0e8
    heights = [i * depth for i in range(100)]

    spectrum = sim_point(temp, surface_pressure, elements, wavelengths, incident_spectrum, heights, depth)

    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.plot(wavelengths, spectrum, linewidth=2)
    ax.set_title('Spectrum Intensity')
    ax.set_xlabel('Wavelength (nm)')
    ax.set_ylabel('Intensity')
    ax.set_xlim(380, 780)
    ax.grid(True)
    plt.show()

calc_and_show()
