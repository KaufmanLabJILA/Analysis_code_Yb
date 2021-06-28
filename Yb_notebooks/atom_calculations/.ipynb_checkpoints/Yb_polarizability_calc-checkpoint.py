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

#171
final_states_1S0 = {
    "6s2_1S0-6s6p_1P1": final_state((751525987.761 + 939.52)*(1e6), 2*np.pi*29.1e6, 1, 1/2),
    "6s2_1S0-6s6p_3P1": final_state((539386561 + 1825.72)*(1e6), 2*np.pi*182.4e3, 1, 1/2),
    "6s2_1S0-6s7p_1P1": final_state(c/(246.50e-9), 0.2151e7, 1, 1/2),
    "6s2_1S0-6p2_3P1": final_state(c/(377.15e-9), 1.233e8, 1, 1/2),
    "6s2_1S0-5d6s2_5252": final_state(c/(257.36e-9), 9.2e7, 1, 1/2),
    "6s2_1S0-5d6s2_5232": final_state(c/(268.65e-9), 2.15e8, 1, 1/2),
    "6s2_1S0-5d6s2_7252": final_state(c/(346.34e-9), 2.627e8, 1, 1/2)
}

final_states_3P0 = {
    "6s6p_3P0-6s7s_3S1": final_state(c/(649.1e-9), 9.6e6, 1, 1/2),
    "6s6p_3P0-5d6s_3D1": final_state(c/(1479.3e-9), 2.0e6, 2, 1/2),
    "6s6p_3P0-6s6d_3D1": final_state(c/(444.71e-9), 0.95e7, 1, 1/2),
    "6s6p_3P0-6p2_3P1": final_state(c/(377.15e-9), 1.233e8, 1, 1/2),
    "6s6p_3P0-6s7d_3D1": final_state(c/(370.48e-9), 1.633e7, 1, 1/2),
    "6s6p_3P0-6s7s_3S1": final_state(c/(650.45e-9), 0.973e7, 1, 1/2)
}

final_states_3P1 = {
    "6s6p_3P1-6s2_1S0": final_state(-(539386561 + 1825.72)*(1e6), 2*np.pi*182.4e3, 0, 1/2),
    "6s6p_3P1-6s7s_3S1": final_state(c/(680.1e-9), 2.7e7, 1, 1/2),
    "6s6p_3P1-5d6s_3D1": final_state(c/(1479.3e-9), 2.0e6, 1, 1/2),
    "6s6p_3P1-6s6d_3D1": final_state(c/(457.60e-9), 0.6186e7, 1, 1/2),
    "6s6p_3P1-6s6d_3D2": final_state(c/(456.93e-9), 1.299e7, 2, 1/2),
    "6s6p_3P1-6p2_3P0": final_state(c/(407.16e-9), 2.66e8, 0, 1/2),
    "6s6p_3P1-6p2_3P1": final_state(c/(386.38e-9), 8.44e7, 1, 1/2),
    "6s6p_3P1-6p2_3P2": final_state(c/(376.13e-9), 8.66e7, 2, 1/2),
    "6s6p_3P1-6s7d_3D1": final_state(c/(379.38e-9), 1.123e7, 1, 1/2),
    "6s6p_3P1-6s7d_1D2": final_state(c/(378.52e-9), 0.686e6, 2, 1/2),
    "6s6p_3P1-6s7s_1S0": final_state(c/(609.89e-9), 3.451e6, 0, 1/2),
    "6s6p_3P1-5d6s6p_7272": final_state(c/(360.58e-9), 2.454e6, 0, 1/2),
    "6s6p_3P1-5d6s6p_7252": final_state(c/(360.58e-9), 0.1081e5, 2, 1/2)
}

final_states_174_3P2 = {
    "6s6p_3P2-6p2_3P2": final_state(c/(402.74e-9), 1.467e8, 0, 1/2),
    "6s6p_3P2-6p2_3P1": final_state(c/(414.51e-9), 1.112e8, 1, 1/2),
    "6s6p_3P2-6s6d_3D2": final_state(c/(496.80e-9), 2.2e6, 2, 1/2),
    "6s6p_3P2-6s6d_3D3": final_state(c/(493.69e-9), 1.08e7, 3, 1/2),
    "6s6p_3P2-6s6d_3D1": final_state(c/(497.59e-9), 2.93e5, 1, 1/2),
    "6s6p_3P2-6s6d_1D2": final_state(c/(491.38e-9), 4.78e5, 2, 1/2),
    "6s6p_3P2-6s7d_3D1": final_state(c/(406.46e-9), 0.637e6, 1, 1/2),
    "6s6p_3P2-6s7d_3D2": final_state(c/(406.45e-9), 0.51e6, 2, 1/2),
    "6s6p_3P2-6s7d_3D3": final_state(c/(405.47e-9), 2.31e7, 3, 1/2),
    "6s6p_3P2-6s10d_3D3": final_state(c/(351.16e-9), 0.16e7, 3, 1/2),
    "6s6p_3P2-6s7s_3S1": final_state(c/(770.16e-9), 0.28e8, 1, 1/2),
}


final_states_174_1P1 = {
    "6s6p_1P1_-6p2_1S0": final_state(c/(430.02e-9), 0.54e8, 0, 1/2),
    "6s6p_1P1_-6s10d_1D2": final_state(c/(431.35e-9), 1.3036e7, 2, 1/2),
    "6s6p_1P1-5d6s6p_7232": final_state(c/(426.53e-9), 0.4433e5, 2, 1/2)
}

#174
final_states_174_1S0 = {
    "6s2_1S0-6s6p_1P1": final_state((751525987.761 + 939.52)*(1e6), 2*np.pi*29.1e6, 1, 0),
    "6s2_1S0-6s6p_3P1": final_state((539386561 + 1825.72)*(1e6), 2*np.pi*182.4e3, 1, 0),
    "6s2_1S0-6s7p_1P1": final_state(c/(246.50e-9), 0.2151e7, 1, 0),
    "6s2_1S0-6p2_3P1": final_state(c/(377.15e-9), 1.233e8, 1, 0),
    "6s2_1S0-5d6s2_5252": final_state(c/(257.36e-9), 9.2e7, 1, 0),
    "6s2_1S0-5d6s2_5232": final_state(c/(268.65e-9), 2.15e8, 1, 0),
    "6s2_1S0-5d6s2_7252": final_state(c/(346.34e-9), 2.627e8, 1, 0)
}

final_states_174_3P0 = {
    "6s6p_3P0-6s7s_3S1": final_state(c/(649.1e-9), 9.6e6, 1, 0),
    "6s6p_3P0-5d6s_3D1": final_state(c/(1479.3e-9), 2.0e6, 2, 0),
    "6s6p_3P0-6s6d_3D1": final_state(c/(444.71e-9), 0.95e7, 1, 0),
    "6s6p_3P0-6p2_3P1": final_state(c/(377.15e-9), 1.233e8, 1, 0),
    "6s6p_3P0-6s7d_3D1": final_state(c/(370.48e-9), 1.633e7, 1, 0),
    "6s6p_3P0-6s7s_3S1": final_state(c/(650.45e-9), 0.973e7, 1, 0)
}

final_states_174_3P1 = {
    "6s6p_3P1-6s2_1S0": final_state(-(539386561 + 1825.72)*(1e6), 2*np.pi*182.4e3, 0, 0),
    "6s6p_3P1-6s7s_3S1": final_state(c/(680.1e-9), 2.7e7, 1, 0),
    "6s6p_3P1-5d6s_3D1": final_state(c/(1479.3e-9), 2.0e6, 1, 0),
    "6s6p_3P1-6s6d_3D1": final_state(c/(457.60e-9), 0.6186e7, 1, 0),
    "6s6p_3P1-6s6d_3D2": final_state(c/(456.93e-9), 1.299e7, 2, 0),
    "6s6p_3P1-6p2_3P0": final_state(c/(407.16e-9), 2.66e8, 0, 0),
    "6s6p_3P1-6p2_3P1": final_state(c/(386.38e-9), 8.44e7, 1, 0),
    "6s6p_3P1-6p2_3P2": final_state(c/(376.13e-9), 8.66e7, 2, 0),
    "6s6p_3P1-6s7d_3D1": final_state(c/(379.38e-9), 1.123e7, 1, 0),
    "6s6p_3P1-6s7d_1D2": final_state(c/(378.52e-9), 0.686e6, 2, 0),
    "6s6p_3P1-6s7s_1S0": final_state(c/(609.89e-9), 3.451e6, 0, 0),
    "6s6p_3P1-5d6s6p_7272": final_state(c/(360.58e-9), 2.454e6, 0, 0),
    "6s6p_3P1-5d6s6p_7252": final_state(c/(360.58e-9), 0.1081e5, 2, 0)
}

final_states_174_3P2 = {
    "6s6p_3P2-6p2_3P2": final_state(c/(402.74e-9), 1.467e8, 0, 0),
    "6s6p_3P2-6p2_3P1": final_state(c/(414.51e-9), 1.112e8, 1, 0),
    "6s6p_3P2-6s6d_3D2": final_state(c/(496.80e-9), 2.2e6, 2, 0),
    "6s6p_3P2-6s6d_3D3": final_state(c/(493.69e-9), 1.08e7, 3, 0),
    "6s6p_3P2-6s6d_3D1": final_state(c/(497.59e-9), 2.93e5, 1, 0),
    "6s6p_3P2-6s6d_1D2": final_state(c/(491.38e-9), 4.78e5, 2, 0),
    "6s6p_3P2-6s7d_3D1": final_state(c/(406.46e-9), 0.637e6, 1, 0),
    "6s6p_3P2-6s7d_3D2": final_state(c/(406.45e-9), 0.51e6, 2, 0),
    "6s6p_3P2-6s7d_3D3": final_state(c/(405.47e-9), 2.31e7, 3, 0),
    "6s6p_3P2-6s10d_3D3": final_state(c/(351.16e-9), 0.16e7, 3, 0),
    "6s6p_3P2-6s7s_3S1": final_state(c/(770.16e-9), 0.28e8, 1, 0),
}

final_states_174_1P1 = {
    "6s6p_1P1_-6p2_1S0": final_state(c/(430.02e-9), 0.54e8, 0, 0),
    "6s6p_1P1_-6s10d_1D2": final_state(c/(431.35e-9), 1.3036e7, 2, 0),
    "6s6p_1P1-5d6s6p_7232": final_state(c/(426.53e-9), 0.4433e5, 2, 0)
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
