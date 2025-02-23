from enum import Enum, auto
from typing import Tuple
import math
import numpy as np
from numpy.typing import NDArray

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

def calc_elem_dist(height: float, temp: float, surface_pressure: float, elements: list[Element]) -> list[Tuple[Element, float]]:
    pressures = [pressure(temp, height, elem.molecular_mass, GRAVITY, surface_pressure) for elem in elements]
    total_pressure = sum(pressures)
    return [(elem, pressure / total_pressure if total_pressure != 0.0 else 0.0) for elem, pressure in zip(elements, pressures)]

def planck_func(wavelength: float, temp: float) -> float:
    h = 6.626e-34  # Planck constant (J s)
    c = 3e8  # Speed of light (m/s)
    k_B = 1.380649e-23  # Boltzmann constant (J/K)
    return 2 * h * c**2 / wavelength**5 / (math.exp(h * c / (wavelength * k_B * temp)) - 1)

def calc_spectra(dist_by_layer: list[list[Tuple[Element, float]]], wavelengths: NDArray[np.float64], depth: float, temp: float) -> Tuple[NDArray[np.float64], NDArray[np.float64]]:
    emission_spectra = np.zeros((len(dist_by_layer), len(wavelengths)))
    transmitted_spectra = np.zeros((len(dist_by_layer), len(wavelengths)))

    for i, dist in enumerate(dist_by_layer):
        for j, wavelength in enumerate(wavelengths):
            absorption_coeff = sum(frac * elem.absorption_cross_section(wavelength) for elem, frac in dist)
            optical_depth = absorption_coeff * depth
            transmission = math.exp(-optical_depth)
            absorption = 1 - transmission
            emission_spectra[i, j] = absorption * planck_func(wavelength, temp)
            transmitted_spectra[i, j] = transmission

    return emission_spectra, transmitted_spectra

def sim_point(temp: float, surface_pressure: float, elements: list[Tuple[Element, float]], wavelengths: NDArray[np.float64], incident_spectrum: NDArray[np.float64], heights: NDArray[np.float64], depth: float) -> NDArray[np.float64]:
    assert sum(cont for _, cont in elements) == 1.0
    assert len(wavelengths) == len(incident_spectrum)

    dist_by_layer = [calc_elem_dist(height, temp, surface_pressure, list(map(lambda x: x[0], elements))) for height in heights]

    emission_spectra, transmitted_spectra = calc_spectra(dist_by_layer, wavelengths, depth, temp)

    assert(len(emission_spectra) == len(transmitted_spectra))

    #  For each layer, the spectrum that's transmitted through all layers above
    transmitted_above = np.zeros((len(emission_spectra), len(wavelengths)))
    for i, emission_spectrum in enumerate(emission_spectra):
        transmitted_above[i] = emission_spectrum.copy()
        for j in range(i, len(emission_spectra)):
            transmitted_above[i] *= transmitted_spectra[j]

    # This is how much of the stars spectrum reaches the surface
    incident_surface_spectrum = incident_spectrum.copy()
    for transmission_spectrum in transmitted_spectra:
        incident_surface_spectrum *= transmission_spectrum

    spectrum = incident_surface_spectrum
    # spectrum = np.zeros(len(wavelengths))
    for layer in transmitted_above:
        spectrum += layer

    assert len(spectrum) == len(wavelengths)

    return spectrum
