from enum import Enum, auto
from typing import Tuple
import math
import numpy as np
from numpy.typing import NDArray
from astropy.io import fits

hdul = fits.open('data/wavelength.fits')

h2_data = np.loadtxt('data/H2_data.txt', skiprows=3)
he_data = np.loadtxt('data/He_data.txt', skiprows=3)
ti_data = np.loadtxt('data/Ti_data.txt', skiprows=3)
h2_160_data = np.loadtxt('data/1H2-16O_data.txt', skiprows=3)
wl=[0.5,5.0]

wavelength_to_idx_cache = {}

def find_nearest_index(wavelength, data):
    if wavelength in wavelength_to_idx_cache:
        return wavelength_to_idx_cache[wavelength]
    else:
        index = np.argmin(np.abs(data - wavelength))
        wavelength_to_idx_cache[wavelength] = index
        return index

class Element(Enum):
    HELIUM = auto()
    H2 = auto()
    TI = auto()
    H2_160 = auto()

    @property
    def molecular_mass(self) -> float:
        # In g/mol
        if self == Element.HELIUM:
            return 4.002603
        if  self == Element.H2:
            return 2.016
        if self == Element.TI:
            return 63.87
        if self == Element.H2_160:
            return 18.01528
        assert False

    def absorption_cross_section(self, wavelength: float) -> float:
        idx = find_nearest_index(wavelength / 1000.0, hdul[0].data)

        if self == Element.HELIUM:
            return he_data[idx]
        if self == Element.H2:
            return h2_data[idx]
        if self == Element.TI:
            return ti_data[idx]
        if self == Element.H2_160:
            return h2_160_data[idx]
        assert False


def scale_height(temp: float, mean_molecular_mass: float, gravity: float) -> float:
    # H = RT / Mg where R is the molar gas constant and M is the mean moral mass
    R = 8.31446
    return R * temp / (mean_molecular_mass * gravity)

def pressure(temp: float, height: float, mean_molecular_mass: float, gravity: float, surface_pressure: float) -> float:
    scale_height_ = scale_height(temp, mean_molecular_mass, gravity)
    return surface_pressure * math.exp(-height / scale_height_)

GRAVITY = 9.81

def calc_elem_dist(height: float, temp: float, surface_pressure: float, elements: NDArray) -> NDArray:
    pressures = [pressure(temp, height, elem.molecular_mass, GRAVITY, surface_pressure) for elem in elements]
    total_pressure = sum(pressures)
    return np.array([pressure / total_pressure if total_pressure != 0.0 else 0.0 for pressure in pressures])

def planck_func(wavelength: float, temp: float) -> float:
    h = 6.626e-34  # Planck constant (J s)
    c = 3e8  # Speed of light (m/s)
    k_B = 1.380649e-23  # Boltzmann constant (J/K)
    return 2 * h * c**2 / wavelength**5 / (math.exp(h * c / (wavelength * k_B * temp)) - 1)

def calc_spectra(dist_by_layer: NDArray, elements: NDArray, wavelengths: NDArray[np.float64], depth: float, temp: float) -> Tuple[NDArray[np.float64], NDArray[np.float64]]:
    emission_spectra = np.zeros((len(dist_by_layer), len(wavelengths)))
    transmitted_spectra = np.zeros((len(dist_by_layer), len(wavelengths)))

    absorption_cross_sections = np.array([elem.absorption_cross_section(wavelength) * 1e30 for elem in elements for wavelength in wavelengths])
    print(absorption_cross_sections)

    planck_by_wavelength = np.array([planck_func(wavelength, temp) for wavelength in wavelengths])

    for i, dist in enumerate(dist_by_layer):
        for j, wavelength in enumerate(wavelengths):
            absorption_coeff = 0.0
            for k in range(len(dist)):
                absorption_coeff += dist[k] * absorption_cross_sections[k * len(wavelengths) + j]
            optical_depth = absorption_coeff * depth
            transmission = math.exp(-optical_depth)
            absorption = 1 - transmission
            emission_spectra[i, j] = absorption * planck_by_wavelength[j]
            transmitted_spectra[i, j] = transmission

    return emission_spectra, transmitted_spectra

def sim_point(temp: float, surface_pressure: float, elements: list[Tuple[Element, float]], wavelengths: NDArray[np.float64], incident_spectrum: NDArray[np.float64], heights: NDArray[np.float64], depth: float) -> NDArray[np.float64]:
    assert len(wavelengths) == len(incident_spectrum)

    _elements = np.array(list(map(lambda x: x[0], elements)))
    dist_by_layer = np.array([calc_elem_dist(height, temp, surface_pressure, _elements) for height in heights])

    emission_spectra, transmitted_spectra = calc_spectra(dist_by_layer, _elements, wavelengths, depth, temp)

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

    spectrum = np.zeros(len(wavelengths))
    for layer in transmitted_above:
        spectrum += layer

    assert len(spectrum) == len(wavelengths)

    return spectrum

ELEMENT_CONTENTS = [(Element.H2, 0.6), (Element.HELIUM, 0.15), (Element.H2_160, 5e-3), (Element.TI, 5e-4)]
TEMPERATURE = 2200.0
SURFACE_PRESSURE = 130e5
