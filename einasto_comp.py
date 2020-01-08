"""
Created on 2020

@author: Eduardo Vitral

"""
###############################################################################
#
# December 2019, Paris
#
# This file compares the volume density of Einasto (1965) and the one from
# the Sersic profile (1963).
#
# Documentation is provided on Vitral & Mamon, 2020a. 
# If you have any further questions please email vitral@iap.fr
#
###############################################################################

import numpy as np
from scipy.integrate import quad
import matplotlib.pyplot as plt
from scipy.special import gamma, gammaincinv

#import AsinhNorm

plt.rcParams['font.family'] = 'serif'
plt.rcParams['font.serif'] = ['Computer Modern Roman'] \
    + plt.rcParams['font.serif']
    
from matplotlib import rc
rc('font',**{'family':'sans-serif','sans-serif':['Helvetica']})
## for Palatino and other serif fonts use:
#rc('font',**{'family':'serif','serif':['Palatino']})
rc('text', usetex=True)
    
#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
#---------------------------------------------------------------------------
"Initial definitions"
#---------------------------------------------------------------------------

# Give the grid ranges
R_left  = -3  # Initial value of log(r/R_e)
R_right = 3   # Final value of log(r/R_e)
steps_R = 100 # Number of log(r/R_e) bins

m_left  = np.log10(0.5) # Initial value of log(n)
ms_right = np.log10(10)  # Final value of log(n) for the Sersic profile
me_right = np.log10(20)  # Final value of log(n) for the Einasto profile
steps_ns = 1000          # Number of log(n) bins for the Sersic index
steps_ne = 2000          # Number of log(n) bins for the Einasto index

# Value that will be used to limit calculations on the domain where 
# dens(r) > dens(R_e) / BIG
BIG  = 1e30

#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
#---------------------------------------------------------------------------
"Numerical calculation of nu(r/R_e)"
#---------------------------------------------------------------------------

def integrand(x,n,r0) :
    """
    integrand of the density
    
    """
    return np.exp(- b(n) * x**(1/n)) * (x**(1/n - 1)) / np.sqrt(x**2 - r0**2)

def getNU(x,n) :
    """
    constructs the grid of density values
    
    """
    # Index of the fit r_{1/2}/R_e = SUM_i=0,3 ai * log^i(n)
    ai = np.asarray([1.32491, 0.0545396, -0.0286632, 0.0035086])
    # Checks if x = r/R_e is an array
    if (np.isscalar(x)) :
        x = np.asarray([x])
    # Checks if n is an array
    if (np.isscalar(n)) :
        n = np.asarray([n])
    nu = np.zeros((len(x),len(n))) 
    for i in range(0,len(x)) :
        for j in range(0,len(n)) :
            
            # x_0 is the result of r_{1/2}/R_e for the current Sersic index
            x_0 = ai[0] + ai[1]*np.log10(n[j]) + ai[2]*np.log10(n[j])**2 + \
                   ai[3]*np.log10(n[j])**3
            # computes X-critical (c.f. Vitral & Mamon 2020a)
            x_crit = (9*np.log(10)/b(n[j]))**n[j]
            norm   = 2*b(n[j])**(2*n[j]+1) / (np.pi * n[j]**2 * gamma(2*n[j]))
            if (x_crit <= x[i]) :
                integ   = quad(integrand, x[i], np.inf, args=(n[j],x[i]),
                              epsrel=1e-4,epsabs=0,limit=1000)[0]
         
                nu[i,j] = norm * integ
            else :
                int1   = quad(integrand, x[i], x_crit, args=(n[j],x[i]),
                              epsrel=1e-4,epsabs=0,limit=1000)[0]
                int2   = quad(integrand, x_crit, np.inf, args=(n[j],x[i]),
                              epsrel=1e-4,epsabs=0,limit=1000)[0]
                nu[i,j] = norm * (int1 + int2)
            # Now we compute the value of rho_tilde(r_{1/2})
            if (x_crit <= x_0) :
                integ   = quad(integrand, x_0, np.inf, args=(n[j],x_0),
                              epsrel=1e-4,epsabs=0,limit=1000)[0]
         
                nu_0 = norm * integ
            else :
                int1   = quad(integrand, x_0, x_crit, args=(n[j],x_0),
                              epsrel=1e-4,epsabs=0,limit=1000)[0]
                int2   = quad(integrand, x_crit, np.inf, args=(n[j],x_0),
                              epsrel=1e-4,epsabs=0,limit=1000)[0]
                nu_0 = norm * (int1 + int2)
            # We get the ratio rho(r)/rho(r_{1/2})   
            nu[i,j] = nu[i,j]/nu_0
            
    return nu

def b(n) :
    """
    Formula from Ciotti & Bertin 1999, (CB99)
    
    """
    b = 2*n - 1/3 + 4/(405*n) + 46/(25515*n**2) + 131/(1148175*n**3) - \
        2194697/(30690717750*n**4)
    return b

def getEinasto(x,n) :
    """
    constructs the grid of density values for the Einasto (1965) profile
    
    """
    # Index of the fit log(R_e/r_{1/2}) = SUM_i=0,4 ai * log^i(n)
    ai = np.asarray([0.00273979, 0.440575, 0.51598, -0.142506, 0.767049])
    nu = np.zeros((len(x),len(n))) 
    for i in range(0,len(x)) :
        for j in range(0,len(n)) :
            # ratio is R_e/r_{-2}
            ratio =   10**(ai[0] + ai[1]*np.log10(n[j]) + \
                      ai[2]*np.log10(n[j])**2 + ai[3]*np.log10(n[j])**3 + \
                      ai[4]*np.log10(n[j])**4)
            z = x[i] * ratio * (2*n[j])**(n[j]) # z = r/a
            solution = gammaincinv(3*n[j],0.5)
            nu[i,j]  = np.exp(-(z)**(1/n[j]) + solution)
            
    return nu

#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
#---------------------------------------------------------------------------
"rms info"
#---------------------------------------------------------------------------

def getRMS(EIN,SER) :
    """
    Give the grid of Einasto and Sersic deprojected profiles, this function 
    calculates the best correspondance of n_Sersic and n_Einasto, as well as
    the RMS of the logarithm of this ratio.
    
    """
    n    = ms # Sersic indices
    m    = me # Einasto indices
    info = np.zeros((len(n),2))
    for i in range(0,len(n)) :
        rms       = np.zeros(len(m))
        for j in range(0,len(m)) :
            diff = np.log10(SER[:,i]/EIN[:,j])
            not_zero1 = np.where(np.logical_not(np.isnan(diff**2)))
            not_zero2 = np.where(np.logical_not(np.isinf(diff**2)))
            not_zero  = np.intersect1d(not_zero1,not_zero2)
            # Calculates the RMS
            rms[j]       = np.sqrt(np.nanmean(diff[not_zero]**2))
 
        idx       = np.nanargmin(rms)
        info[i,0] = m[idx]
        info[i,1] = rms[idx]
        
    return info

#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
#---------------------------------------------------------------------------
"main"
#---------------------------------------------------------------------------
eta    = np.logspace(R_left,R_right,num = steps_R) # eta = r/Re

ms     = np.logspace(m_left,ms_right,num = steps_ns) # ms: Sersic index

me     = np.logspace(m_left,me_right,num = steps_ne) # me: Einasto index

rho_Ein = getEinasto(eta,me)

#rho_Ser = getNU(eta,ms) 
#np.save('nu_rh', rho_Ser)
rho_Ser = np.load('nu_rh.npy')

rms_info = getRMS(rho_Ein,rho_Ser)

#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
#---------------------------------------------------------------------------
"Fitting procedure"
#---------------------------------------------------------------------------

fig, ax1 = plt.subplots(figsize=(7, 5))
ax1.tick_params(labeltop=False, labelright=True, top = True, right = True, \
                axis='both', which='major', labelsize=18, direction="in", \
                length = 8)
ax1.tick_params(labeltop=False, labelright=True, top = True, right = True, \
                axis='both', which='minor', labelsize=18, direction="in", \
                length = 4)

ax1.set_ylabel('$n_{\mathrm{Einasto}}$',fontsize=18,color='red')
# we already handled the x-label with ax1
ax1.loglog(np.linspace(0.5,30,10),np.linspace(0.5,30,10),color='red',
           label=r'$n_{\mathrm{Einasto}}=n_{\mathrm{S\acute{e}rsic}}$', 
           linestyle='--',lw=1,dashes=(4, 4))
ax1.loglog(ms,rms_info[:,0],color='red',lw=2)
ax1.tick_params(axis='y', labelcolor='red')
ax1.set_xlabel(r'$n_{\mathrm{S\acute{e}rsic}}$',fontsize=18)
ax1.set_xlim([ms[0],ms[len(ms)-1]])
ax1.set_yticks(np.asarray([0.5,1,2,5,10]))
ax1.set_yticklabels(['$0.5$','$1$','$2$','$5$','$10$'])
ax1.legend(loc='best',prop={'size': 18})
ax1.grid(color='pink')
ax2 = ax1.twinx()  # instantiate a second axes that shares the same x-axis
ax2.tick_params(labeltop=False, labelright=True, top = True, right = True, \
                axis='both', which='major', labelsize=18, direction="in", \
                length = 8)
ax2.tick_params(labeltop=False, labelright=True, top = True, right = True, \
                axis='both', which='minor', labelsize=18, direction="in", \
                length = 4)
ax2.loglog(ms,rms_info[:,1], color='blue')
ax2.set_ylabel(r'$\mathrm{rms} \ \log{\left( ' + \
               r'\displaystyle{\frac{\rho_{\mathrm{S\acute{e}rsic}}(r)}' + \
               r'{\rho_{\mathrm{S\acute{e}rsic}}(r_{\mathrm{h}})} \, ' + \
               r'\frac{\rho_{\mathrm{Einasto}}(r_{\mathrm{h}})}' + \
               r'{\rho_{\mathrm{Einasto}}(r)}}\right)}$',
               color='blue',fontsize=18)
ax2.tick_params(axis='y', labelcolor='blue')
ax2.set_xticks(np.asarray([0.5,1,2,5,10]))
ax2.set_xticklabels(['$0.5$','$1$','$2$','$5$','$10$'])
ax2.grid(color='lightskyblue')
plt.minorticks_on()
fig.tight_layout()  # otherwise the right y-label is slightly clipped
plt.savefig('Einasto.pdf', format = 'pdf', bbox_inches="tight")
plt.show()

