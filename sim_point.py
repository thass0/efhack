from enum import Enum, auto
from typing import Tuple
from dataclasses import dataclass

class Element(Enum):
    IRON = auto()
    HELIUM = auto()
    OXYGEN = auto()

def sim_point(temp: int, pressure: int, elements: list[Tuple[Element, float]]) -> [int, float]:
    assert sum(cont for _, cont in elements) == 1.0
    return [0, 0.0]

