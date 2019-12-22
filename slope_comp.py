"""
Created on 2019

@author: Eduardo Vitral

"""
###############################################################################
#
# December 2019, Paris
#
# This file compares the model proposed by Lima Neto, Gerbal & Márquez 1999
# with the numerical deprojection of the Sérsic model (Sérsic 1963; 
# Sersic 1968).
#
# Documentation is provided on Vitral & Mamon, 2020a. 
# If you have any further questions please email vitral@iap.fr
#
###############################################################################

import numpy as np
from scipy.integrate import quad
import matplotlib.pyplot as plt
from scipy.special import gamma
import itertools

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
m_right = np.log10(10)  # Final value of log(n)
steps_n = 50            # Number of log(n) bins

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
    # Checks if x = r/R_e is an array
    if (np.isscalar(x)) :
        x = np.asarray([x])
    # Checks if n is an array
    if (np.isscalar(n)) :
        n = np.asarray([n])
    nu = np.zeros((len(x),len(n))) 
    for i in range(0,len(x)) :
        for j in range(0,len(n)) :
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
    return nu

#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
#---------------------------------------------------------------------------
"Sersic deprojections formulae"
#---------------------------------------------------------------------------

def ReadFile (file_name) :
    """
    Reads a text file by dividing into words
    
    """
    with open(file_name,'r') as f:
        words = [word.strip() for word in f]
    f.close()
    return(words)

def getCoeff (file_name) :
    """
    Gets coefficients of the polynomial fit (begins at line 5)
    
    """
    words = ReadFile(file_name)
    coeff = list()
    for i in range (5,len(words)) :
        coeff.append(float(words[i]))
    return np.asarray(coeff)

def pPS(n) :
    """
    Formula from Prugniel & Simien 1997, (PS97)
    
    """
    p = 1 - 1.188/(2*n) + 0.22/(4*n**2)
    return p

def pLN(n) :
    """
    Formula from Lima Neto, Gerbal & Márquez 1999, (LGM99)
    
    """
    p = 1 - 0.6097/n + 0.05563/(n**2)
    return p

def b(n) :
    """
    Formula from Ciotti & Bertin 1999, (CB99)
    
    """
    b = 2*n - 1/3 + 4/(405*n) + 46/(25515*n**2) + 131/(1148175*n**3) - \
        2194697/(30690717750*n**4)
    return b

def getNUm(x,n,model,*args) :
    """
    Constructs the density grid for a certain model
    
    """
    nu = np.zeros((len(x),len(n))) 
    for i in range(0,len(x)) :
        for j in range(0,len(n)) :
            if (model == 'LN') :
                p = pLN(n[j])
            if (model == 'PS') :
                p = pPS(n[j])
            try:
                p
            except NameError:
                print("You did not give a valid model")
                return
            norm    = b(n[j])**(n[j]*(3-p)) / (n[j] * gamma(n[j]*(3-p)))
            nu[i,j] = norm * np.exp(- b(n[j]) * x[i]**(1/n[j])) * x[i]**(-p)
            
    return nu

def factor(i) :
    """
    Computes a normalization factor
    
    """
    n   = m1[i]
    norm = (gamma(n*(3-pLN(n))) * n) / (b(n)**(n*(3-pLN(n))))
    fac = np.exp(b(n) * eta[:]**(1/n)) * norm
    return fac


#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
#---------------------------------------------------------------------------
"Generates data"
#---------------------------------------------------------------------------

# Creates logspaced array of r/R_e
eta    = np.logspace(R_left,R_right,num = steps_R)
# Sam thing for m: Sersic index
m      = np.logspace(m_left,m_right,num = steps_n)

m1      = np.asarray([0.5,1,2,4,8]) + 0.25# m: Sersic index

# Gets the density for a numerical computation and
# Lima Neto, Gerbal & Márquez 1999 approximation
nu_num1 = getNU(eta,m1)
nu_LN1  = getNUm(eta,m1,'LN')

#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
#---------------------------------------------------------------------------
"Plotting routine"
#---------------------------------------------------------------------------

fig, axes = plt.subplots(figsize=(6, 6))

colors = ['red', 'b', 'limegreen', 'deepskyblue', 'orange']
cc = itertools.cycle(colors)
plot_lines = []
plt.tick_params(labeltop=False, labelright=False, top = True, right = True, \
                axis='both', which='major', labelsize=18, direction="in", \
                length = 8)
plt.tick_params(labeltop=False, labelright=False, top = True, right = True, \
                axis='both', which='minor', labelsize=18, direction="in", \
                length = 4)
for i in range(0,len(colors)):

    d1 = factor(i)*nu_num1[:,i]
    d2 = factor(i)*nu_LN1[:,i]
  
    c = next(cc)
    if (i == 0) :
        axes.loglog(eta,d1, lw=2.5, color='black')
        axes.loglog(eta,d2,lw=1.5,linestyle='--',dashes=(5, 5),color='black')
        
    axes.loglog(eta,d1, lw=4, color=c)
    axes.loglog(eta,d2,lw=2, color=c)
    axes.loglog(eta,d2,lw=2,linestyle='--',dashes=(5, 5),color='black')
    
lines = axes.get_lines()
legend1 = plt.legend([lines[i] for i in [0,1]], ["$\mathrm{Numerical}$", 
                     "$\mathrm{LGM}$"], loc=3,
                     prop={'size': 18})
legend2 = plt.legend([lines[i] for i in [2,5,8,11,14]], ["$n = " + \
                     str(m1[i]) + "$" for i in range(5)], loc=1, 
                     prop={'size': 18})
axes.add_artist(legend1)
axes.add_artist(legend2)
plt.xlabel(r'$r/R_{\mathrm{e}}$', fontsize = 18)
plt.ylabel(r"$\widetilde{\rho} \, \times \, $" + \
           r"$\displaystyle{\frac{n \, \Gamma[n(3-p)]}{b^{n(3-p)}}} \,$" + \
           r" $\times \, \exp [\, b \, (r/R_{\mathrm{e}})^{1/n}]$", 
           fontsize = 18)
plt.xlim([eta[0],eta[len(eta)-1]])
plt.grid()
plt.minorticks_on()
plt.savefig('PowerLaw.pdf', format = 'pdf', bbox_inches="tight") 
plt.show()
