import numpy as np
import scipy.constants as sc
from sympy.physics.wigner import wigner_6j


""" 
constants
"""
pi = np.pi
c = sc.c
epsilon_0 = sc.epsilon_0
hbar = sc.hbar
h = sc.h
aud = 8.4783536255e-30

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
    def __init__(self, f, A_J, J, I):
        self.w = 2 * pi * f
        self.A_J = A_J # Einstein A-coefficient [s^-1] (A_J)
        self.J = J
        self.I = J

def GammaJgJe(k, Jg, Je, M):
    w0 = 2*np.pi*c*(k*1e2)
    return ((w0**3)/(3*pi*epsilon_0*hbar*(c**3))) * ( (2*Jg+1)/(2*Je+1) ) * ((M * aud)**2)
    

#174
final_states_174_1S0 = {
    "6s2_1S0-6s6p_1P1": final_state((751525987.761 + 939.52)*(1e6), 2*np.pi*29.1e6, 1, 0),
    "6s2_1S0-6s6p_3P1": final_state((539386561 + 1825.72)*(1e6), 2*np.pi*182.4e3, 1, 0),
    "6s2_1S0-6s7p_3P1": final_state(c*(38174.17e2), GammaJgJe(38174.17, 0, 1, 0.08), 1, 0),
    "6s2_1S0-6s8p_3P1": final_state(c*(43659.38e2), GammaJgJe(43659.38, 0, 1, 0.01), 1, 0),
    "6s2_1S0-6s7p_1P1": final_state(c*(40563.97e2), GammaJgJe(40563.97, 0, 1, 0.65), 1, 0),
    "6s2_1S0-6s8p_1P1": final_state(c*(44017.60e2), GammaJgJe(44017.60, 0, 1, 0.21), 1, 0)
#     "6s2_1S0-6p2_3P1": final_state(c/(377.15e-9), 1.233e8, 1, 0),
#     "6s2_1S0-5d6s2_5252": final_state(c/(257.36e-9), 9.2e7, 1, 0),
#     "6s2_1S0-5d6s2_5232": final_state(c/(268.65e-9), 2.15e8, 1, 0),
#     "6s2_1S0-5d6s2_7252": final_state(c/(346.34e-9), 2.627e8, 1, 0)
}

final_states_174_3P0 = {
    "6s6p_3P0-6s7s_3S1": final_state(c*(32694.692-17288.439)*(1e2), GammaJgJe(32694.692-17288.439, 0, 1, 1.95), 1, 0),
    "6s6p_3P0-5d6s_3D1": final_state(c*(24489.102-17288.439)*(1e2), GammaJgJe(24489.102-17288.439, 0, 1, 2.89), 2, 0),
    "6s6p_3P0-6s6d_3D1": final_state(c*(39808.72-17288.439)*(1e2), GammaJgJe(39808.72-17288.439, 0, 1, 1.84), 1, 0),
    "6s6p_3P0-6p2_3P1": final_state(c*(43805.42-17288.439)*(1e2), GammaJgJe(43805.42-17288.439, 0, 1, 1.97), 1, 0),
    "6s6p_3P0-6s7d_3D1": final_state(c*(44311.38-17288.439)*(1e2), GammaJgJe(44311.38-17288.439, 0, 1, 2.10), 1, 0),
    "6s6p_3P0-6s8s_3S1": final_state(c*(41615.04-17288.439)*(1e2), GammaJgJe(41615.04-17288.439, 0, 1, 0.62), 1, 0)
}

final_states_174_3P1 = {
    "6s6p_3P1-6s2_1S0": final_state(-(539386561 + 1825.72)*(1e6), 2*np.pi*182.4e3, 1, 0),
    "6s6p_3P1-5d6s_3D1": final_state(c*(24489.102-17992.007)*(1e2), GammaJgJe(24489.102-17992.007, 1, 1, 2.51), 1, 0),
    "6s6p_3P1-5d6s_3D2": final_state(c*(24751.948-17992.007)*(1e2), GammaJgJe(24751.948-17992.007, 1, 2, 4.35), 2, 0),
    "6s6p_3P1-5d6s_1D1": final_state(c*(27677.665-17992.007)*(1e2), GammaJgJe(27677.665-17992.007, 1, 1, 0.453), 1, 0),
    "6s6p_3P1-6s6d_3D1": final_state(c*(39808.72-17992.007)*(1e2), GammaJgJe(39808.72-17992.007, 1, 1, 1.57), 1, 0),
    "6s6p_3P1-6s6d_3D2": final_state(c*(39838.04-17992.007)*(1e2), GammaJgJe(39838.04-17992.007, 1, 2, 2.78), 2, 0),
    "6s6p_3P1-6s6d_1D2": final_state(c*(40061.51-17992.007)*(1e2), GammaJgJe(40061.51-17992.007, 1, 2, 0.53), 2, 0),
    "6s6p_3P1-6s7d_3D1": final_state(c*(44311.38-17992.007)*(1e2), GammaJgJe(44311.38-17992.007, 1, 1, 2.47), 1, 0),
    "6s6p_3P1-6s7d_3D2": final_state(c*(44313.05-17992.007)*(1e2), GammaJgJe(44313.05-17992.007, 1, 2, 1.67), 2, 0),
    "6s6p_3P1-6s7d_1D2": final_state(c*(44357.60-17992.007)*(1e2), GammaJgJe(44357.60-17992.007, 1, 2, 0.86), 2, 0),
    "6s6p_3P1-6s7s_3S1": final_state(c*(32694.692-17992.007)*(1e2), GammaJgJe(32694.692-17992.007, 1, 1, 3.46), 1, 0),
    "6s6p_3P1-6s7s_1S0": final_state(c*(34350.65-17992.007)*(1e2), GammaJgJe(34350.65-17992.007, 1, 0, 0.243), 0, 0),
    "6s6p_3P1-6s8s_3S1": final_state(c*(41615.04-17992.007)*(1e2), GammaJgJe(41615.04-17992.007, 1, 1, 1.00), 1, 0),
    "6s6p_3P1-6s8s_1S0": final_state(c*(41939.90-17992.007)*(1e2), GammaJgJe(41939.90-17992.007, 1, 0, 0.30), 0, 0),
    "6s6p_3P1-6p2_3P0": final_state(c*(42436.91-17992.007)*(1e2), GammaJgJe(42436.91-17992.007, 1, 0, 2.59), 0, 0),
    "6s6p_3P1-6p2_3P1": final_state(c*(43805.42-17992.007)*(1e2), GammaJgJe(43805.42-17992.007, 1, 1, 0.18), 1, 0),
    "6s6p_3P1-6p2_3P2": final_state(c*(44760.37-17992.007)*(1e2), GammaJgJe(44760.37-17992.007, 1, 2, 2.92), 2, 0),
}

#these are from Porsev paper https://journals.aps.org/pra/pdf/10.1103/PhysRevA.60.2781
final_states_174_1P1 = {
    "6s6p_1P1-6s2_1S0": final_state(-c*(25068.222)*(1e2), 2*np.pi*29.1e6, 1, 0),
    "6s6p_1P1-5d6s_3D1": final_state(c*(24489.102-25068.222)*(1e2), GammaJgJe(24489.102-25068.222, 1, 1, 0.27), 1, 0),
    "6s6p_1P1-5d6s_3D2": final_state(c*(24751.948-25068.222)*(1e2), GammaJgJe(24751.948-25068.222, 1, 2, 0.32), 2, 0),
    "6s6p_1P1-5d6s_1D2": final_state(c*(27677.665-25068.222)*(1e2), GammaJgJe(27677.665-25068.222, 1, 2, 3.6), 2, 0),
    "6s6p_1P1-6s7s_3S1": final_state(c*(32694.692-25068.222)*(1e2), GammaJgJe(32694.692-25068.222, 1, 1, 0.74), 1, 0),
    "6s6p_1P1-6s7s_1S0": final_state(c*(34350.65-25068.222)*(1e2), GammaJgJe(34350.65-25068.222, 1, 1, 4.31), 1, 0)
}


def M(i, f):
    """ 
    square of reduced dipole matrix element from A_J coefficient
    |<J||d||J'>|^2
    """
    return (
           (3. * pi * epsilon_0 * hbar * c**3. / f.w**3.)
           * f.A_J 
           * ((2 * f.J + 1) / (2 * i.J + 1))
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
                         -alpha_tensor(i, f_dict)(w)
                         * ((3.*i.m**2 - i.F * (i.F + 1.)) / (i.F * (2. * i.F - 1.)))
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
