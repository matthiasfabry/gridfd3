import numpy as np

au2km = 1.495978707e8  # (km)
pc2km = 3.085677581e13  # (km)
m_sun = 1.9885e30  # (kg)
r_sun = 6.957e5  # (km)
G = 6.67430e-20  # (km3 kg-1 s-2)
deg2rad = np.pi / 180
rad2deg = 180 / np.pi
mas2rad = 1e-3 / 3600 * deg2rad
day2sec = 86400
rad2mas = rad2deg * 3600 * 1e3


def primary_mass(e, k1, k2, p, sini):
    """
    Calculates the mass of the primary body of the system
    :return: mass of the primary (Solar Mass)
    """
    return np.power(1 - e ** 2, 1.5) * (k1 + k2) ** 2 * k2 * p * day2sec / (2 * np.pi * G * sini ** 3) / m_sun


def secondary_mass(e, k1, k2, p, sini):
    """
    Calculates the mass of the secondary body of the system
    :return: mass of the secondary (in Solar Mass)
    """
    return np.power(1 - e ** 2, 1.5) * (k1 + k2) ** 2 * k1 * p * day2sec / (2 * np.pi * G * sini ** 3) / m_sun


def total_mass(e, k1, k2, p, sini):
    return np.power(1 - e ** 2, 1.5) * (k1 + k2) ** 3 * p * day2sec / (2 * np.pi * G * sini ** 3) / m_sun
