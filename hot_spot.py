#!/usr/bin/env python
"""
Neutron star hot spot pulse profile
"""
from TOV import *
import matplotlib
import matplotlib.pyplot as plt
from matplotlib import rc
import scipy.constants as cst
from scipy.integrate import trapz as integ
from pynverse import inversefunc
from scipy.interpolate import interp1d
from scipy.misc import derivative as dd
import numpy as np
import math

def Psi(sigma, a, b, r):
    psi = np.zeros(len(sigma))
    r_ = np.copy(r)
    dr = 0.1*(r_[1]-r_[0])
    R = r_[0]
    r_[0] = R+dr
    for i in range(len(sigma)):
        v1 = np.sqrt(a[0]*b[0])/(R**2)*(-2*np.sqrt(1-sigma[i]**2*a[0]/r_[0]**2)/r_[0]**2)*dr
        v2 = integ(np.sqrt(a*b)/(r_**2*np.sqrt(1-a*sigma[i]**2/r_**2)),r_)
        psi[i] = sigma[i]*(v1+v2)
    return psi

def Shapiro(sigma, a, b, r):
    shapiro = np.zeros(len(sigma))
    r_ = np.copy(r)
    dr = 0.1*(r_[1]-r_[0])
    R = r_[0]
    r_[0] = R+dr
    for i in range(len(sigma)):
        v1 = (np.sqrt(b[0]/a[0])*(-2*np.sqrt(1-sigma[i]**2*a[0]/r_[0]**2)/r_[0]**2)-np.sqrt(b[0]/a[0]))*dr
        v2 = integ(np.sqrt(b/a)/(np.sqrt(1-a*sigma[i]**2/r_**2)),r_) -integ(np.sqrt(b/a),r_)
        shapiro[i] = (v1+v2)/cst.c
    return shapiro

def ComputeAlphaAndDerivative(cPsi, sAlpha, cPsi_f_sAlpha):
    sAlpha_ = np.zeros(len(cPsi))
    DCPsi_DSAlpha = np.zeros(len(cPsi))
    for i in range(len(cPsi)):
        idx = len(np.where(cPsi_f_sAlpha<psi[i])[-1])
        if idx != len(cPsi_f_sAlpha):
            a = (sAlpha[idx+1]-sAlpha[idx])/(cPsi_f_sAlpha[idx+1]-cPsi_f_sAlpha[idx])
            b = sAlpha[idx]-sAlpha[idx]*a
            sAlpha_[i]= psi[i]*a+b
            DCPsi_DSAlpha[i] = a
    return sAlpha_, DCPsi_DSAlpha

def ComputeFlux(iota,gamma,t0,omega,a,b,r,R,phi_R):
    # Compute Psi(alpha)
    alpha = np.linspace(0,np.pi/2,100)
    sigma = R*np.sin(alpha)/np.sqrt(a[0])
    f_psi_alpha = Psi(sigma, a, b, r)
    t = Shapiro(sigma, a, b, r)
    # Compute flux
    psi = np.arccos(np.cos(iota)*np.cos(gamma)+np.sin(iota)*np.sin(gamma)*np.cos(omega*t0))
    id_nvis = np.where(psi>f_psi_alpha[-1])
    psi[id_nvis] = f_psi_alpha[-1]
    f_1 = interp1d(f_psi_alpha, alpha, kind='cubic')
    alpha_2 = f_1(psi)
    df_1 = interp1d(f_psi_alpha, np.gradient(alpha)/np.gradient(f_psi_alpha), kind='cubic')
    dAlpha_dPsi = df_1(psi)
    beta = R*omega*np.sin(gamma)/np.sqrt(a[0])
    cosXi = -np.sin(alpha_2)*np.sin(iota)*np.sin(omega*t0)/np.sin(psi)
    delta = np.sqrt(1-(beta/cst.c)**2)/(1-(beta/cst.c)*cosXi)
    Flux = phi_R*delta**5*a[0]*np.cos(alpha_2)*np.sin(alpha_2)*dAlpha_dPsi/np.sin(psi)
    Flux[id_nvis] = 0
    # Compute observed time
    sigma = R*np.sin(alpha_2)/np.sqrt(a[0])
    delta_t = Shapiro(sigma, a, b, r)
    t_obs = t0+delta_t-delta_t[0]
    return Flux, t_obs

def ComputeFluxTot(iota,gamma,t0,omega,a,b,r,R,phi_R):
    Flux, t_obs = ComputeFlux(iota,gamma,t0,omega,a,b,r,R,phi_R)
    Flux_opposite, t_obs = ComputeFlux(iota,np.pi-gamma,t0,omega,a,b,r,R,phi_R)
    return Flux_opposite+Flux, t_obs

def PlotFlux():
    # initialize variable
    PhiInit = 1
    PsiInit = 0
    radiusMax_in = 40000
    radiusMax_out = 100000000
    Npoint = 10000
    log_active = True
    colorlist = {}
    labels = {}
    linestyles = {}
    rho0 = 5000
    rhoInit = rho0
    iota = np.pi/4
    gamma = np.pi/4
    f = 400
    omega = 2*np.pi*f
    t0 = np.linspace(0.0000000001,4*10**(-3),100)
    for i in range(2):
        if i==2:
            lag_mode = 0
            dilaton_active = True
            colorlist[2] = 'red'
            labels[2] = 'ER ($L_m=T$)'
            linestyles[2] = '--'
        elif i==0:
            lag_mode = 0
            dilaton_active = False
            colorlist[0] = 'k'
            labels[0] = 'GR'
            linestyles[0] = '-'
        if i==1:
            lag_mode = 1
            dilaton_active = True
            colorlist[1] = 'orangered'
            labels[1] = 'ER ($L_m=-c^2\\rho$)'
            linestyles[1] = '--'
        if i==3:
            lag_mode = 2
            dilaton_active = True
            colorlist[3] = 'orange'
            labels[3] = 'ER ($L_m=P$)'
            linestyles[3] = '--'
        # Compute NS exterior space-time
        tov = TOV(rhoInit, PsiInit, PhiInit, radiusMax_in, radiusMax_out, Npoint, lag_mode, dilaton_active, log_active, 1)
        tov.ComputeTOV()
        R = tov.radiusStar
        r = tov.r_ext
        a = tov.g_tt_ext
        b = tov.g_rr_ext
        phi_R = tov.phiStar
        # Compute Psi(alpha) and plot psi-alpha and delta_t-alpha
        alpha = np.linspace(0,np.pi/2,100)
        sigma = R*np.sin(alpha)/np.sqrt(a[0])
        f_psi_alpha = Psi(sigma, a, b, r)
        plt.figure(1)
        plt.plot(alpha,f_psi_alpha,color=colorlist[i])
        plt.xlabel('$\\alpha$ [rad]')
        plt.ylabel('$\\psi$ [rad]')
        t = Shapiro(sigma, a, b, r)
        plt.figure(2)
        plt.plot(alpha,(t-t[0])*1000,color=colorlist[i])
        plt.xlabel('$\\alpha$ [rad]')
        plt.ylabel('t [ms]')
        # Compute flux
        psi = np.arccos(np.cos(iota)*np.cos(gamma)+np.sin(iota)*np.sin(gamma)*np.cos(omega*t0))
        id_nvis = np.where(psi>f_psi_alpha[-1])
        psi[id_nvis] = f_psi_alpha[-1]
        f_1 = interp1d(f_psi_alpha, alpha, kind='cubic')
        alpha_2 = f_1(psi)
        df_1 = interp1d(f_psi_alpha, np.gradient(alpha)/np.gradient(f_psi_alpha), kind='cubic')
        dAlpha_dPsi = df_1(psi)
        beta = R*omega*np.sin(gamma)/np.sqrt(a[0])
        cosXi = -np.sin(alpha_2)*np.sin(iota)*np.sin(omega*t0)/np.sin(psi)
        delta = np.sqrt(1-(beta/cst.c)**2)/(1-(beta/cst.c)*cosXi)
        Flux = phi_R*delta**5*a[0]*np.cos(alpha_2)*np.sin(alpha_2)*dAlpha_dPsi/np.sin(psi)
        Flux[id_nvis] = 0
        # Compute observed time
        sigma = R*np.sin(alpha_2)/np.sqrt(a[0])
        delta_t = Shapiro(sigma, a, b, r)
        t_obs = t0+delta_t-delta_t[0]
        plt.figure(3)
        plt.plot(alpha,f_psi_alpha,color=colorlist[i])
        plt.plot(alpha_2,psi,'o')
        plt.xlabel('$\\alpha$')
        plt.ylabel('$\\psi$')
        plt.figure(4)
        plt.plot(t_obs*1000,Flux,color=colorlist[i],label=labels[i],linestyle=linestyles[i])
        plt.xlabel('observer time [ms]')
        plt.ylabel('Noralized flux')
        plt.legend()
        plt.title('$\\iota=${:.2f}, $\gamma=${:.2f}, $\\rho_0=$ {:.1e} $Mev/fm^3$'.format(iota,gamma,rho0))
    plt.show()
