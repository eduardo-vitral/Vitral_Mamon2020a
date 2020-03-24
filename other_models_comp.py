"""
Created on 2020

@author: Eduardo Vitral

"""
###############################################################################
#
# December 2019, Paris
#
# This file compares the volume density of Plummer (1911), Jaffe (1983), 
# Hernquist (1990) and Navarro et al. (1996) with the Sersic profile (1963).
#
# Documentation is provided on Vitral & Mamon, 2020a. 
# If you have any further questions please email vitral@iap.fr
#
###############################################################################

import numpy as np
from scipy.integrate import quad
import matplotlib.pyplot as plt
from scipy.special import gamma

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
m_right = np.log10(8)   # Final value of log(n)
steps_n = 9             # Number of log(n) bins

#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
#---------------------------------------------------------------------------
"Numerical calculation of nu(r/R_e)"
#---------------------------------------------------------------------------

def integrand(x,n,r0) :
    """
    integrand of the density
    
    """
    return np.exp(- b(n) * x**(1/n)) * (x**(1/n - 1)) / np.sqrt(x**2 - r0**2)

def getNU(y,n) :
    """
    constructs the grid of density values
    
    """
    # Checks if y = r/r_{1/2} is an array
    if (np.isscalar(y)) :
        y = np.asarray([y])
    nu = np.zeros(len(y))
    
    # Index of the fit r_{1/2}/R_e = SUM_i=0,3 ai * log^i(n)
    ai = np.asarray([1.32491, 0.0545396, -0.0286632, 0.0035086])

    # summ is the result of r_{1/2}/R_e
    summ = ai[0] + ai[1]*np.log10(n) + ai[2]*np.log10(n)**2 + \
           ai[3]*np.log10(n)**3
           
    x   = summ * y # x = r/R_e
    x_0 = summ     # x_0 = r_{1/2}/R_e
    # computes X-critical (c.f. Vitral & Mamon 2020a)
    x_crit = (9*np.log(10)/b(n))**n
    norm   = 2*b(n)**(2*n+1) / (np.pi * n**2 * gamma(2*n))
        
    for i in range(0,len(x)) :
        if (x_crit <= x[i]) :
            integ   = quad(integrand, x[i], np.inf, args=(n,x[i]),
                          epsrel=1e-4,epsabs=0,limit=1000)[0]
     
            nu[i] = norm * integ
        else :
            int1   = quad(integrand, x[i], x_crit, args=(n,x[i]),
                          epsrel=1e-4,epsabs=0,limit=1000)[0]
            int2   = quad(integrand, x_crit, np.inf, args=(n,x[i]),
                          epsrel=1e-4,epsabs=0,limit=1000)[0]
            nu[i] = norm * (int1 + int2)
    # Now we compute the value of rho_tilde(r_{1/2})
    if (x_crit <= x_0) :
        integ   = quad(integrand, x_0, np.inf, args=(n,x_0),
                      epsrel=1e-4,epsabs=0,limit=1000)[0]
 
        nu_0 = norm * integ
    else :
        int1   = quad(integrand, x_0, x_crit, args=(n,x_0),
                      epsrel=1e-4,epsabs=0,limit=1000)[0]
        int2   = quad(integrand, x_crit, np.inf, args=(n,x_0),
                      epsrel=1e-4,epsabs=0,limit=1000)[0]
        nu_0 = norm * (int1 + int2)
    # We get the ratio rho(r)/rho(r_{1/2})  
    return (nu/nu_0)

def b(n) :
    """
    Formula from Ciotti & Bertin 1999, (CB99)
    
    """
    b = 2*n - 1/3 + 4/(405*n) + 46/(25515*n**2) + 131/(1148175*n**3) - \
        2194697/(30690717750*n**4)
    return b

def getPlummer(x) :
    """
    Formula from [rho(r)/rho(r_{1/2})], for the Plummer (1911) profile
    
    """
    y    = x*(1+2**(1/3))/np.sqrt(3)  # y = r/a
    nu   = (1 + y**2)**(-2.5)
    nu_0 = (1 + ((1+2**(1/3))/np.sqrt(3))**2)**(-2.5)
    
    return (nu/nu_0)

def getJaffe(x):
    """
    Formula from [rho(r)/rho(r_{1/2})], for the Jaffe (1983) profile
    
    """
    nu   = 1/(x**2 * (1+x)**2)
    nu_0 = 1/4
    return (nu/nu_0)

def getHernquist(x) :
    """
    Formula from [rho(r)/rho(r_{1/2})], for the Hernquist (1990) profile
    
    """
    y    = x*(1+np.sqrt(2))  # y = r/a
    nu   = 1/(y * (1+y)**3)
    nu_0 = 1/((1+np.sqrt(2)) * (1 + (1+np.sqrt(2)))**3)
    
    return (nu/nu_0)

def getNFW(x,c) :
    """
    Formula from [rho(r)/rho(r_{1/2})], for the Navarro et al. (1996) profile
    
    """
    # concentration values
    conc = np.asarray([3.5, 5, 7, 10, 14, 20])
    # ratio r_{1/2}/a for each respective concentration
    fact = np.asarray([1.69797, 2.2166, 2.82098, 3.60561, 4.50533, 5.65844])
    idx  = np.argmin(np.abs(conc-c))
    y    = x * fact[idx]
    nu   = 1/(y * (1+y)**2)
    nu_0 = 1/(fact[idx] * (1+fact[idx])**2)
    return (nu/nu_0)

#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
#---------------------------------------------------------------------------
"main"
#---------------------------------------------------------------------------
eta    = np.logspace(R_left,R_right,num = steps_R) # eta = r/r_{1/2}

m      = np.logspace(m_left,m_right,num = steps_n) # m: Sersic index

# gets the density grid for different models
rho_Plumm = getPlummer(eta)
rho_Jaffe = getJaffe(eta)
rho_Hernq = getHernquist(eta)
rho_NFW35 = getNFW(eta,3.5)
rho_NFW05 = getNFW(eta,5)
rho_NFW07 = getNFW(eta,7)
rho_NFW10 = getNFW(eta,10)
rho_NFW14 = getNFW(eta,14)
rho_NFW20 = getNFW(eta,20)

model = [rho_NFW35,rho_NFW05,rho_NFW07,rho_NFW10,rho_NFW14,rho_NFW20,
         rho_Plumm,rho_Hernq,rho_Jaffe]

colorNFW = 'yellowgreen' # NFW color for the plots

# Selected colors for each model
colors   = [colorNFW,colorNFW,colorNFW,colorNFW,colorNFW,colorNFW,
          'magenta','red','blue']
# Selected labels for each model
labels = [r'$\mathrm{NFW} \ (3.5 < r_{\mathrm{max}}/a < 20)$',
          r'$\mathrm{Plummer}$',r'$\mathrm{Hernquist}$',r'$\mathrm{Jaffe}$']
# Selected colors for each Sersic index
colors_s = ['saddlebrown','teal','grey','firebrick','darkviolet',
            'deepskyblue','hotpink','orange','green']

fig, axes = plt.subplots(figsize=(7.3, 6))
plt.tick_params(labeltop=False, labelright=False, top = True, right = True, \
                axis='both', which='major', labelsize=16, direction="in", \
                length = 8)
plt.tick_params(labeltop=False, labelright=False, top = True, right = True, \
                axis='both', which='minor', labelsize=16, direction="in", \
                length = 4)

for i in range(0,len(model)) :
    if (i < 6) :
        plt.loglog(eta,model[i],color=colors[i],lw=1.5)
    else :
        plt.loglog(eta,model[i],color=colors[i],lw=3)
for i in range(0,len(m)) :
    sersic = getNU(eta,m[i])
    if (i==0) :
        plt.loglog(eta,sersic, color='black', 
                   linestyle='--',dashes=(5, 5),lw=1.5)
    plt.loglog(eta,sersic, linestyle='--', color=colors_s[i],
               dashes=(5, 5), lw=1.5)
lines = axes.get_lines()
legend1 = plt.legend([lines[i] for i in range(5,len(lines)-len(m)-1)], labels, 
                     loc=3, prop={'size': 15})
labels_sersic = [r'$\mathrm{S\acute{e}rsic}$']
for i in range(len(m)) :
    labels_sersic.append("$n= " + str(round(m[i],2)) + "$")

legend2 = plt.legend([lines[i] for i in range(len(lines)-len(m)-1,len(lines))],
                     labels_sersic,
                     loc=1, 
                     prop={'size': 15})
axes.add_artist(legend1)
axes.add_artist(legend2)    
plt.xlim([eta[0],eta[len(eta)-1]])
plt.ylim([1e-6,1e6])
plt.ylabel(r'$\rho (r) \, / \, ' + \
          r'\rho (r_{\mathrm{h}}) $',fontsize=21)
plt.xlabel(r'$r \, / \, r_{\mathrm{h}}$',fontsize=21)
#plt.grid()
plt.savefig(r'OtherModels.pdf', format = 'pdf', bbox_inches="tight")
plt.show()

