import numpy as np
import scipy as sp
import scipy.constants
from sympy.physics.wigner import wigner_6j


""" 
constants
"""
pi = np.pi
c = sp.constants.c
epsilon_0 = sp.constants.epsilon_0
hbar = sp.constants.hbar
h = sp.constants.h


""" 
generic upper and lower states
"""

class initial_state(object):
    def __init__(self, J, I, F, m):
        self.J = J
        self.I = I
        self.F = F
        self.m = m

class final_state(object):
    def __init__(self, WN, A_J, J, I):
        self.w = 2 * pi * c * WN * 1e2
        self.A_J = A_J * 1e6 # Einstein A-coefficient [s^-1] (A_J)
        self.J = J
        self.I = J

"""
numbers from 
Magic wavelengths for terahertz clock transitions
Zhou et al. PRA 81 012115 (2010)
"""

final_states_1S0 = {
              "5s5p 1P1": final_state(21698.48, 190.01, 1., 9/2.),
              "5s6p 1P1": final_state(34098.44,  1.87, 1., 9/2.),
              "5s7p 1P1": final_state(38906.9,  5.32, 1., 9/2.),
              "5s8p 1P1": final_state(41172.15, 14.9, 1., 9/2.),
              "4d5p 1P1": final_state(41184.47, 12, 1., 9/2.),
              "5s9p 1P1": final_state(42462.36, 11.6, 1., 9/2.),
              "5s10p 1P1": final_state(43327.94, 7.6, 1., 9/2.),
              "5s11p 1P1": final_state(43938.26, 4.88, 1., 9/2.),

              "5s5p 3P1": final_state(14504.35, .0469, 1, 9/2.),
             }

final_states_3P0 = {
              "5s6s 3S1" : final_state(14721.275, 10.226, 1., 9/2.),
              "5s7s 3S1" : final_state(23107.193, 1.402, 1., 9/2.),
              "5s8s 3S1" : final_state(26443.920, 0.954, 1., 9/2.),
              "5s9s 3S1" : final_state(28133.68, 0.525, 1., 9/2.),
              "5s10s 3S1": final_state(29110.080, 0.32, 1., 9/2.),

              "5p5p 3P1" : final_state(21082.618, 41.492, 1., 9/2.),

              "5s4d 3D1" : final_state(3841.536, 0.29, 1., 9/2.),
              "5s5d 3D1" : final_state(20689.423, 35.732, 1., 9/2.),
              "5s6d 3D1" : final_state(25368.383, 14.303, 1., 9/2.),
              "5s7d 3D1" : final_state(27546.88, 8.223, 1., 9/2.),
              "5s8d 3D1" : final_state(28749.18, 4.920, 1., 9/2.),
              "5s9d 3D1" : final_state(29490.28, 3.184, 1., 9/2.), #Mistake on wavelenth here was 24940, should be 29490
             }

final_states_3P1 = {
              "5s6s 3S1": final_state(14534.444, 29.526, 1., 9/2.),
              "5s7s 3S1": final_state(22920.362, 4.106, 1., 9/2.),
              "5s8s 3S1": final_state(26257.089,  2.803, 1., 9/2.),
              "5s9s 3S1": final_state(27946.849, 1.543, 1., 9/2.),
              "5s10s 3S1": final_state(28923.249, 0.943, 1., 9/2.),
              
              "5p5p 3P0": final_state(20689.119, 117.64, 0., 9/2.),
              "5p5p 3P1": final_state(20895.787, 30.297, 1., 9/2.),
              "5p5p 3P2": final_state(21170.317, 31.509, 2., 9/2.),

              "5s4d 3D1": final_state(3654.705, 0.187, 1., 9/2.),
              "5s4d 3D2": final_state(3714.444, 0.354, 2., 9/2.),
              "5s5d 3D1": final_state(20502.592, 26.08, 1., 9/2.),
              "5s5d 3D2": final_state(20517.664, 47.049, 2., 9/2.),
              "5s6d 3D1": final_state(25181.552, 10.492, 1., 9/2.),
              "5s6d 3D2": final_state(25186.488, 18.897, 2., 9/2.),
              "5s7d 3D1": final_state(27360.049, 6.043, 1., 9/2.),
              "5s7d 3D2": final_state(27364.969, 10.833, 2., 9/2.),
              "5s8d 3D1": final_state(28562.349, 3.619, 1., 9/2.),
              "5s8d 3D2": final_state(28565.959, 6.517, 2., 9/2.),
              "5s9d 3D1": final_state(29303.449, 2.342, 1., 9/2.),
              "5s9d 3D2": final_state(29303.449, 4.216, 2., 9/2.),
              }

final_states_3P2 = {
              "5s6s 3S1": final_state(14140.232, 45.314, 1, 9/2.),
              "5s7s 3S1": final_state(22526.15, 6.495, 1, 9/2.),
              "5s8s 3S1": final_state(25862.877, 4.464, 1, 9/2.),
              "5s9s 3S1": final_state(27552.637, 2.464, 1, 9/2.),
              "5s10s 3S1": final_state(28529.037, 1.508, 1, 9/2.),
              
              "5p5p 3P1": final_state(20501.575, 47.69, 1, 9/2.),
              "5p5p 3P2": final_state(20776.105, 89.343, 2, 9/2.),

              "5s4d 3D1": final_state(3260.493, 0.009, 1, 9/2.),
              "5s4d 3D2": final_state(3320.232, 0.084, 2, 9/2.),
              "5s4d 3D3": final_state(3420.704, 0.368, 3, 9/2.),
              "5s5d 3D1": final_state(20108.38, 1.64, 1, 9/2.),
              "5s5d 3D2": final_state(20123.452, 14.796, 2, 9/2.),
              "5s5d 3D3": final_state(20146.492, 59.39, 3, 9/2.),
              "5s6d 3D1": final_state(24787.34, 0.667, 1, 9/2.),
              "5s6d 3D2": final_state(24792.276, 6.008, 2, 9/2.),
              "5s6d 3D2": final_state(24804.591, 24.069, 3, 9/2.),
              "5s7d 3D1": final_state(26965.837, 0.386, 1, 9/2.),
              "5s7d 3D2": final_state(26970.757, 3.473, 2, 9/2.),
              "5s7d 3D3": final_state(26976.337, 13.902, 3, 9/2.),
              "5s8d 3D1": final_state(28168.137, 0.231, 1, 9/2.),
              "5s8d 3D2": final_state(28171.747, 2.083, 2, 9/2.),
              "5s8d 3D3": final_state(28176.207, 8.337, 3, 9/2.),
              "5s9d 3D1": final_state(28909.237, 0.15, 1, 9/2.),
              "5s9d 3D2": final_state(28909.237, 1.35, 2, 9/2.),
              "5s9d 3D3": final_state(28914.037, 5.401, 3, 9/2.),
              }

def M(i, f):
    """ 
    square of reduced dipole matrix element from A_J coefficient
    |<J||d||J'>|^2
    """
    return (
           3. * pi * epsilon_0 * hbar * c**3. / f.w**3.
           * f.A_J 
           * (2 * f.J + 1) / (2 * i.J + 1)
           )

""" 
polarizability expressions taken from Steck's Quantum Optics
eqn. 7.483
"""

def alpha_scalar(i, f_dict):
    return lambda w: sum([
                         2. / 3.
                         * f.w * M(i, f)
                         / (f.w**2 - w**2)
                     for f in f_dict.values()]) / hbar

def alpha_vector(i, f_dict):
    return lambda w: sum([
                         (-1)**float(-2*i.J - f.J - i.F - i.I + 1)
                         * np.sqrt(6 * i.F * (2 * i.F + 1) / (i.F + 1))
                         * (2 * i.J + 1)
                         * wigner_6j(1, 1, 1, i.J, i.J, f.J)
                         * wigner_6j(i.J, i.J, 1, i.F, i.F, i.I)
                         * f.w * M(i, f)
                         / (f.w**2 - w**2)
                     for f in f_dict.values()]) / hbar

def alpha_tensor(i, f_dict):
    return lambda w: sum([
                         (-1)**float(-2*i.J - f.J - i.F - i.I + 1)
                         * np.sqrt(
                             40. * i.F * (2. * i.F + 1.) * (2. * i.F - 1.) 
                             / 3. / (i.F + 1.) / (2. * i.F + 3.)
                         )
                         * (2 * i.J + 1)
                         * wigner_6j(1, 1, 2, i.J, i.J, f.J)
                         * wigner_6j(i.J, i.J, 2, i.F, i.F, i.I)
                         * f.w * M(i, f)
                         / (f.w**2 - w**2)
                     for f in f_dict.values()]) / hbar

def scalar_shift(i, f_dict, E_field):
    return lambda w:  (
                      -alpha_scalar(i, f_dict)(w)*
                      np.linalg.norm(E_field)**2
                      )

def vector_shift(i, f_dict, E_field):
    if i.F:
        return lambda w:  (
                          -alpha_vector(i, f_dict)(w)*
                          i.m/i.F*
                          (1j*np.cross(E_field, E_field.conj())[2]).real
                          )
    else:
        return lambda x: 0.

def tensor_shift(i, f_dict, E_field):
    if i.F:
        return lambda w: (
                         alpha_tensor(i, f_dict)(w)
                         * (3.*i.m**2 - i.F * (i.F + 1.)) / (i.F * (2. * i.F - 1.))
                         * (3.*np.absolute(E_field[2])**2 - np.linalg.norm(E_field)**2)/2.
                         )
    else:
        return lambda x: 0.

def total_shift(i, f_dict, E_field):
    """ sums scalar vector and tensor shifts for given electric field configuration """
    return lambda w: (
                     scalar_shift(i, f_dict, E_field)(w) 
                     + vector_shift(i, f_dict, E_field)(w) 
                     + tensor_shift(i, f_dict, E_field)(w)
                     )

def total_shift_sum(i, f_dict, E_fields):
    """ sums total shift over list of electric field configurations """
    return lambda w: sum([
                         scalar_shift(i, f_dict, E_field)(w) 
                         + vector_shift(i, f_dict, E_field)(w) 
                         + tensor_shift(i, f_dict, E_field)(w) 
                     for E_field in E_fields])
