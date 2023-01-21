#!/usr/bin/env python
from hot_spot import *
import numpy as np

#-------------------------------------------------------------------------------
# TOV Solver parameters
eqs_mode = 1
PhiInit = 1
PsiInit = 0
radiusMax_in = 40000
radiusMax_out = 100000000
Npoint = 10000
log_active = False
lag_mode = 1
# Hot spot flux parameters
iota = np.pi/4
gamma = np.pi/4
f = 400
omega = 2*np.pi*f
t0 = np.linspace(0.0000000001,4*10**(-3),100)

# Radius mass relation ---------------------------------------------------------
for dilaton_active in [True, False]:
    mass = []
    rho = np.logspace(np.log(100)/np.log(10),np.log(8000)/np.log(10),100)
    for i_rho in rho:
        print(i_rho,'-----------------------------------------------------------')
        tov = TOV(i_rho, PsiInit, PhiInit, radiusMax_in, radiusMax_out, Npoint, lag_mode, dilaton_active, log_active, eqs_mode)
        tov.ComputeTOV()
        mass.append(tov.massStar)
    if eqs_mode == 1 and dilaton_active == True:
        colorlist = 'orangered'
        labels = 'ER (AP4)'
        linestyles = '-'
    elif eqs_mode == 1 and dilaton_active == False:
        colorlist = 'black'
        labels = 'GR (AP4)'
        linestyles = '--'
    plt.figure(1)
    plt.plot(rho,np.array(mass)/massSun,color=colorlist,label=labels,linestyle=linestyles)
    plt.axhline(y = 1.4)
    plt.xlabel('rho')
    plt.ylabel('mass $M/M_\u2609$', fontsize=12)
    plt.legend()

    # Hot spot flux parameters -----------------------------------------------------
    if eqs_mode == 1 and dilaton_active == True:
        rhoInit = 569
        colorlist = 'orangered'
        labels = 'ER (AP4)'
        linestyles = '-'
    elif eqs_mode == 1 and dilaton_active == False:
        rhoInit = 631
        colorlist = 'black'
        labels = 'GR (AP4)'
        linestyles = '--'
    tov = TOV(rhoInit, PsiInit, PhiInit, radiusMax_in, radiusMax_out, Npoint, lag_mode, dilaton_active, log_active, eqs_mode)
    tov.ComputeTOV()
    R = tov.radiusStar
    print(tov.massStar/massSun)
    r = tov.r_ext
    a = tov.g_tt_ext
    b = tov.g_rr_ext
    phi_R = tov.phiStar
    Flux_tot, t_obs = ComputeFluxTot(iota,gamma,t0,omega,a,b,r,R,phi_R)
    Flux, t_obs = ComputeFlux(iota,gamma,t0,omega,a,b,r,R,phi_R)
    Flux_op, t_obs = ComputeFlux(iota,np.pi-gamma,t0,omega,a,b,r,R,phi_R)
    plt.figure(2)
    #plt.plot(t_obs*1000,Flux,color=colorlist,label=labels,linestyle=linestyles)
    #plt.plot(t_obs*1000,Flux_op,color=colorlist,label=labels,linestyle=linestyles)
    plt.plot(t_obs*1000,Flux_tot,color=colorlist,label=labels,linestyle=linestyles)
    plt.xlabel('Observer time [s]')
    plt.ylabel('Total flux normalized')
    plt.legend()
plt.show()
