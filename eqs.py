#!/usr/bin/env python
"""
Equation of state
"""
import scipy.constants as cst
import numpy as np
import math

c2 = cst.c**2
k = 1.475*10**(-3)*(cst.fermi**3/(cst.eV*10**6))**(2/3)*c2**(5/3)

def F_0(x):
    return 1/(np.exp(x)+1)

def F_0_p(x):
    return -np.exp(x)/(np.exp(x)+1)**2

def F_1(x, a, b, c, d):
    X = c*(d-x)
    A = (a+b*x)
    B = F_0(X)
    return A*B

def F_1_p(x, a, b, c, d):
    X = c*(d-x)
    A = (a+b*x)
    B = F_0(X)
    Ap = b
    Bp = -c*F_0_p(X)
    return Ap*B+A*Bp

def F_2(x, a, b, c, d, e, f):
    return (a+b*x+c*x**3)/(1+d*x)*F_0(e*(x-f))

def F_2_p(x, a, b, c, d, e, f):
    return ((b+2*c*x**2)*(1+d*x)-d*(a+b*x+c*x**3))/((1+d*x)**2)*F_0(e*(x-f)) \
    + (a+b*x+c*x**3)/(1+d*x)*F_0_p(e*(x-f))

def eqs_SLy(rho, coef):
    xi = np.log10(rho*(1000*100**(-3)))
    zeta = F_2(xi, coef[0], coef[1], coef[2], coef[3], coef[4], coef[5]) \
    +F_1(xi, coef[6], coef[7], coef[8], coef[9]) \
    +F_1(xi, coef[10], coef[11], coef[12], coef[13]) \
    +F_1(xi, coef[14], coef[15], coef[16], coef[17])
    P = (100**(-2)/cst.dyn)*10**(zeta)/100
    dzeta_dxi = F_2_p(xi, coef[0], coef[1], coef[2], coef[3], coef[4], coef[5]) \
    +F_1_p(xi, coef[6], coef[7], coef[8], coef[9]) \
    +F_1_p(xi, coef[10], coef[11], coef[12], coef[13]) \
    +F_1_p(xi, coef[14], coef[15], coef[16], coef[17])
    dP_drho = P/rho*dzeta_dxi
    return P, dP_drho

def PEQS(rho, option_eqs):
    if option_eqs == 0:
        P = k*rho**(5/3)
        dP_drho = (5/3)*k*rho**(5/3-1)
    elif option_eqs == 1:
        a1 = 6.22
        a2 = 6.121
        a3 = 0.005925
        a4 = 16.326
        a5 = 6.48
        a6 = 11.4971
        a7 = 19.105
        a8 = 0.8923
        a9 = 6.54
        a10 = 11.4950
        a11 = -22.775
        a12 = 1.5707
        a13 = 4.3
        a14 = 14.08
        a15 = 27.8
        a16 = -1.653
        a17 = 1.5
        a18 = 14.67
        if np.isscalar(rho): # Manage rho=0 due to bad behavior of eqs_1 for both rho a scalar and an array
            if rho>0:
                P, dP_drho = eqs_SLy(rho, [a1, a2, a3, a4, a5, a6, a7, a8, a9, a10, a11, a12, a13, a14, a15, a16, a17, a18])
            else:
                P = 0
                dP_drho = 1
        else:
            ind_nz = rho.nonzero() # non zero index
            P = np.zeros(rho.size)
            dP_drho = np.ones(rho.size)
            P_, dP_drho_ = eqs_SLy(rho[ind_nz], [a1, a2, a3, a4, a5, a6, a7, a8, a9, a10, a11, a12, a13, a14, a15, a16, a17, a18])
            P[ind_nz] = P_
            dP_drho[ind_nz] = dP_drho_
    return P, dP_drho
