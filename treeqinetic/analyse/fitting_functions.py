import numpy as np


def exp_decreasing(time, initial_amplitude, damping_coeff):
    return initial_amplitude * np.exp(-damping_coeff * time)


def damped_oscillation(time, initial_amplitude, damping_coeff, angular_frequency, phase_angle):
    return initial_amplitude * np.exp(-damping_coeff * time) * np.cos(2 * np.pi * angular_frequency * time + phase_angle)
