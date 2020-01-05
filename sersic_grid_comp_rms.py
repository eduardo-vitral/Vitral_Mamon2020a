"""
Created on 2019

@author: Eduardo Vitral

"""
###############################################################################
#
# December 2019, Paris
#
# This file tests the polynomial of the ratio between the analytical 
# approximation of Lima Neto, Gerbal & Márquez 1999 and the numerical 
# calculation. It compares it with other models in the literature, such as:
# - Lima Neto, Gerbal & Márquez 1999
# - Prugniel & Simien 1997
# - Emsellem & van de Ven 2008
# - Trujillo et al. 2002
# - Simonneau & Prada 2004
#
# Documentation is provided on Vitral & Mamon, 2020a. 
# If you have any further questions please email vitral@iap.fr
#
###############################################################################

import numpy as np
from scipy.integrate import quad
import matplotlib.pyplot as plt
from math import log10, floor, ceil
from scipy.special import gamma, gammainc, kv
from matplotlib.colors import LinearSegmentedColormap, SymLogNorm
from scipy.interpolate import griddata, splrep, splev

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

# Value that will be used to limit calculations on the domain where 
# dens(r) > dens(R_e) / BIG
BIG  = 1e30

#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
#---------------------------------------------------------------------------
"General functions"
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

def RoundIt (Number) :
    """
    Rounds a number up to its three first significative numbers
    
    """
    if (Number == 0) :
        return 0
    else :
        Rounded = round(Number, -int(floor(log10(abs(Number))))+3)
        return Rounded

def intersect2d(arr1,arr2) :
    """
    Function equivalent to numpy.intersect1d, but for 2D arrays
    arr1 and arr2 the two arrays that will be intersected.
    arr1 and arr2 should have the same shape as the output of 
    numpy.where(condition), where the condition is tested on a 2D numpy array
    
    """
    
    arr1 = np.asarray(arr1).T
    arr2 = np.asarray(arr2).T
    
    inters = list()
    
    for i in range (0,len(arr1)) :
        if((arr1[i] == arr2).all(1).any()) :
            inters.append(arr1[i])

    inters = np.asarray(inters).T
    arr    = (inters[0].astype(int),inters[1].astype(int))     
    
    return arr  

#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
#---------------------------------------------------------------------------
"Mass/density formulae"
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
    nu = np.zeros(len(n))
    for j in range(0,len(n)) :
        # computes X-critical (c.f. Vitral & Mamon 2020a)
        x_crit = (9*np.log(10)/b(n[j]))**n[j]
        norm   = 2*b(n[j])**(2*n[j]+1) / (np.pi * n[j]**2 * gamma(2*n[j]))
        if (x_crit <= x) :
            integ   = quad(integrand, x, np.inf, args=(n[j],x),
                          epsrel=1e-4,epsabs=0,limit=1000)[0]
     
            nu[j] = norm * integ
        else :
            int1   = quad(integrand, x, x_crit, args=(n[j],x),
                          epsrel=1e-4,epsabs=0,limit=1000)[0]
            int2   = quad(integrand, x_crit, np.inf, args=(n[j],x),
                          epsrel=1e-4,epsabs=0,limit=1000)[0]
            nu[j] = norm * (int1 + int2)
    return nu

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

def getMm(x,n,model,*args) :
    """
    Constructs the mass grid for a certain model
    
    """
    M = np.zeros((len(x),len(n))) 
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
            
            M[i,j] = gammainc(n[j]*(3-p), b(n[j]) * x[i]**(1/n[j])) 
            
    return M

def getNUSimm(x,n) :
    """
    Constructs a grid of Simonneau & Prada 2004 (SP04) density
    
    """
    nu = np.zeros((len(x),len(n))) 
    x_j = np.asarray([0.046910,0.230765,0.5,0.769235,0.953090])
    w_j = np.asarray([0.118464,0.239314,0.284444,0.239314,0.118464])
    for i in range(0,len(x)) :
        for j in range(0,len(n)) :
            
            l_j = 1 / (1 - x_j**2)**(1/(n[j]-1))
            p_j = w_j * x_j / np.sqrt(1 - (1 - x_j**2)**(2*n[j]/(n[j]-1))) 
            
            fac1 = 4 * b(n[j])**(2*n[j]+1) / \
                (np.pi * n[j]*(n[j]-1) * gamma(2*n[j]) * x[i]**((n[j]-1)/n[j]))
            sum1 = np.sum(p_j * np.exp(- l_j * b(n[j]) * x[i]**(1/n[j])))
            
            nu[i,j] = fac1 * sum1
    
    return nu

def getNUTru(x,n,int_method,func) :
    """
    Constructs a grid of Trujillo et al. 2002 (T+02) density
    Considers the spherical case where f = alpha = beta = 1
    func stands for the interpolation function to be used (griddata or splev)
    int_method stands for the interpolation method (e.g. cubic)
    
    """
    nu = np.zeros((len(x),len(n)))
    
    m    = np.asarray([0.5,1,1.5,2,2.5,3,3.5,4,4.5,5,6,7,8,9,10])
    bes  = np.asarray([-0.5,0,0.43675,0.47773,0.49231,0.49316,0.49280,0.50325,
                     0.51140,0.52169,0.55823,0.58086,0.60463,0.61483,0.66995])
    p    = np.asarray([1,0,0.61007,0.77491,0.84071,0.87689,0.89914,0.91365,
                       0.92449,0.93279,0.94451,0.95289,0.95904,0.96385,
                       0.96731])
    h1   = np.asarray([0,0,-0.07257,-0.04963,-0.03313,-0.02282,-0.01648,
                       -0.01248,-0.00970,-0.00773,-0.00522,-0.00369,-0.00272,
                       -0.00206,-0.00164])
    h2   = np.asarray([0,0,-0.20048,-0.15556,-0.12070,-0.09611,-0.07919,
                       -0.06747,-0.05829,-0.05106,-0.04060,-0.03311,-0.02768,
                       -0.02353,-0.02053])
    h3   = np.asarray([0,0,0.01647,0.08284,0.14390,0.19680,0.24168,0.27969,
                     0.31280,0.34181,0.39002,0.42942,0.46208,0.48997,0.51325])
    
    if (func == True) :
        int_bes = splev(n,splrep(m, bes))
        int_p   = splev(n,splrep(m, p))
        int_h1  = splev(n,splrep(m, h1))
        int_h2  = splev(n,splrep(m, h2))
        int_h3  = splev(n,splrep(m, h3))
    else :
        int_bes = griddata(m, bes, n, method=int_method)
        int_p   = griddata(m, p, n, method=int_method)
        int_h1  = griddata(m, h1, n, method=int_method)
        int_h2  = griddata(m, h2, n, method=int_method)
        int_h3  = griddata(m, h3, n, method=int_method)
        
    for i in range(0,len(x)) :
        for j in range(0,len(n)) :
                       
            h1i  = int_h1[j]
            h2i  = int_h2[j]
            h3i  = int_h3[j]
            pi   = int_p[j]
            besi = int_bes[j]
            
            Ch  = h1i*(np.log10(x[i]))**2 + h2i*(np.log10(x[i])) + h3i
            fac = 2**((3*n[j]-1)/(2*n[j])) * b(n[j])**(2*n[j]+1) / \
                    (np.pi * n[j]**2 * gamma(2*n[j]))
            Knu = kv(besi, b(n[j]) * x[i]**(1/n[j]))
            nu[i,j] = fac * Knu * x[i]**(pi*(1/n[j] - 1)) / (1 - Ch)
    
    return nu

def getNUEG(x,n,int_method,func) :
    """
    Constructs a grid of Emsellem & van de Ven 2008 (EV08) density
    func stands for the interpolation function to be used (griddata or splev)
    int_method stands for the interpolation method (e.g. cubic)
    
    """
    
    nu = np.zeros((len(x),len(n)))
    
    m    = np.asarray([0.5,0.6,0.7,0.8,0.9,1,1.1,1.2,1.3,1.4,1.5,2,2.5,3,3.5,4,
                       4.5,5,5.5,6,6.5,7,7.5,8,8.5,9,9.5,10])
    bes  = np.asarray([0.5,0.47768,0.44879,0.39831,0.25858,0,0.15502,0.25699,
                       0.30896,0.35245,0.39119,0.51822,0.53678,0.54984,0.55847,
                       0.56395,0.57054,0.57950,0.58402,0.58765,0.59512,0.60214,
                       0.60469,0.61143,0.61789,0.62443,0.63097,0.63694])
    p    = np.asarray([1,0.85417,0.94685,1.04467,2.55052,0,1.59086,
                       1.00670,0.88866,0.83763,0.81030,0.76108,0.83093,0.86863,
                       0.89233,0.90909,0.92097,0.93007,0.93735,0.94332,0.94813,
                       0.95193,0.95557,0.95864,0.96107,0.96360,0.96570,
                       0.96788])
    h0   = np.asarray([0,-0.03567,-0.04808,-0.04315,-0.01879,0,0.00041,0.00069,
                       0.00639,0.01405,0.02294,0.07814,0.13994,0.19278,0.23793,
                       0.27678,0.31039,0.33974,0.36585,0.38917,0.41003,0.42891,
                       0.44621,0.46195,0.47644,0.48982,0.50223,0.51379])
    h1   = np.asarray([0,0.26899, 0.10571,0.01763,-0.39382,0,0.15211,0.05665,
                       0.00933,-0.02791,-0.05876,-0.16720,-0.13033,-0.10455 ,
                       -0.08618,-0.07208,-0.06179,-0.05369,-0.04715,-0.04176,
                       -0.03742,-0.03408,-0.03081,-0.02808,-0.02599,-0.02375,
                       -0.02194,-0.02004])
    h2   = np.asarray([0,-0.09016,-0.06893,-0.04971,-0.08828,0,-0.03341,
                       -0.03964,-0.04456,-0.04775,-0.04984,-0.05381,-0.03570,
                       -0.02476,-0.01789,-0.01333,-0.01028,-0.00812,-0.00653,
                       -0.00534,-0.00444,-0.00376,-0.00319,-0.00274,-0.00238,
                       -0.00207,-0.00182,-0.00160])
    h3   = np.asarray([0,0.03993,0.03363,0.02216,-0.00797,0,0.00899,0.01172,
                       0.01150,0.01026,0.00860,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,
                       0])
   
    if (func == True) :
        int_bes = splev(n,splrep(m, bes))
        int_p   = splev(n,splrep(m, p))
        int_h0  = splev(n,splrep(m, h0))
        int_h1  = splev(n,splrep(m, h1))
        int_h2  = splev(n,splrep(m, h2))
        int_h3  = splev(n,splrep(m, h3))
    else :
        int_bes = griddata(m, bes, n, method=int_method)
        int_p   = griddata(m, p, n, method=int_method)
        int_h0  = griddata(m, h0, n, method=int_method)
        int_h1  = griddata(m, h1, n, method=int_method)
        int_h2  = griddata(m, h2, n, method=int_method)
        int_h3  = griddata(m, h3, n, method=int_method)
        
    for i in range(0,len(x)) :
        for j in range(0,len(n)) :
                        
            h0i  = int_h0[j]
            h1i  = int_h1[j]
            h2i  = int_h2[j]
            h3i  = int_h3[j]
            pi   = int_p[j]
            besi = int_bes[j]
            
            Ch  = h3i*(np.log10(x[i]))**3 + h2i*(np.log10(x[i]))**2 + \
                    h1i*(np.log10(x[i])) + h0i
            fac = 2**((3*n[j]-1)/(2*n[j])) * b(n[j])**(2*n[j]+1) / \
                    (np.pi * n[j]**2 * gamma(2*n[j]))
            Knu = kv(besi, b(n[j]) * x[i]**(1/n[j]))
            nu[i,j] = fac * Knu * x[i]**(pi*(1/n[j] - 1)) / (1 - Ch)
    
    return nu

def getMSimm(x,n) :
    """
    Constructs a grid of Simonneau & Prada 2004 (SP04) mass
    
    """
    M = np.zeros((len(x),len(n))) 
    x_j = np.asarray([0.046910,0.230765,0.5,0.769235,0.953090])
    w_j = np.asarray([0.118464,0.239314,0.284444,0.239314,0.118464])
    for i in range(0,len(x)) :
        for j in range(0,len(n)) :
            
            l_j = 1 / (1 - x_j**2)**(1/(n[j]-1))
            p_j = w_j * x_j / np.sqrt(1 - (1 - x_j**2)**(2*n[j]/(n[j]-1))) 
            
            fac1 = 4 / (np.pi * (n[j]-1) * gamma(2*n[j]))
            sum1 = np.sum(gamma(2*n[j]+1)*gammainc(2*n[j]+1,
                          l_j * b(n[j]) * x[i]**(1/n[j])) * p_j / \
                          l_j**(2*n[j]+1))
            
            M[i,j] = fac1 * sum1
    
    return M

def Polynomial(x,n,params) :
    """
    Computes the polynomial P = SUM a_ij log^i(x) log^j(n), by using the 
    parameters provided in a shape such as saved by the code
    sersic_grid_num.py
    
    """
    P        = np.zeros((len(x),len(n))) 
    n_params = len(params)

    for i in range(0,len(x)) :
    
        for j in range(0,len(n)) :
            
            for k in range(1,n_params+1) :
                
                p     = ceil((-3 + np.sqrt(1 + 8*k))/2)
                x_exp = int(k - 1 - p*(p+1)/2)
                n_exp = int(1 - k + p*(p+3)/2)
                coeff = RoundIt(params[n_params - (k-1) -1])
                
                P[i,j] += coeff * \
                            (np.log10(n[j]))**n_exp * \
                              (np.log10(x[i]))**x_exp
    return P

def RMS(diff) :
    """
    Calculates the root-mean-square of diff
    
    """
    rms = np.sqrt(((diff)**2 / len(diff)).sum())
    return rms

def model_RMS(model,num) :
    """
    Gets the root-mean-square (rms) of the ratio model/numerical
    
    """
    Ratio = model/num
        
    not_zero1 = np.where(np.logical_not(np.isnan(np.log10(Ratio)**2)))
    not_zero2 = np.where(np.logical_not(np.isinf(np.log10(Ratio)**2)))
    not_zero  = intersect2d(not_zero1,not_zero2)
    
    bigger   = np.where(nu_num > nu_1/BIG)
    not_zero = intersect2d(not_zero,bigger)
    
    rms = RMS(np.log10(Ratio)[not_zero])
        
    return rms

#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
#---------------------------------------------------------------------------
"main"
#---------------------------------------------------------------------------

# Creates logspaced array of r/R_e
eta    = np.logspace(R_left,R_right,num = steps_R)
# Sam thing for m: Sersic index
m      = np.logspace(m_left,m_right,num = steps_n)

# Gets the density values for r = R_e
nu_1 = [getNU(1,m)]
for i in range(0,steps_R-1) :
    nu_1 = np.append(nu_1,[getNU(1,m)],axis=0) 

# Gets the polynomial coefficients, as well as the numerical calculation
# provided by the code sersic_grid_num.py
coeff_nu = getCoeff('coeff_1.txt')
coeff_M  = getCoeff('coeff_2.txt')
nu_num   = np.load('dens_num.npy')
M_num    = np.load('mass_num.npy')

# Calculates density for other models in the literature
nu_Sim = getNUSimm(eta,m)
nu_Tru = getNUTru(eta,m,'cubic',True)
nu_EG  = getNUEG(eta,m,'cubic',True)
nu_LN  = getNUm(eta,m,'LN')
nu_PS  = getNUm(eta,m,'PS')

# Calculates mass for other models in the literature
M_Sim = getMSimm(eta,m)
M_LN  = getMm(eta,m,'LN')
M_PS  = getMm(eta,m,'PS')

# Gets the ratio between the model proposed on Vitral & Mamon 2020a
# and the numerical deprojection
LOGM_numRatVM = np.log10(M_LN/M_num) - Polynomial(eta,m,coeff_M)
LOGnu_numRatVM = np.log10(nu_LN/nu_num) - Polynomial(eta,m,coeff_nu)

#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
#---------------------------------------------------------------------------
"Checking the fit"
#---------------------------------------------------------------------------

# Creates personal colormaps in shades of gray, yellow and red
n_bins    = 1000
colors    = [(0,0,50/255),(102/255,178/255,1),(1,1,1),
             (1,102/255,102/255),(50/255,0,0)]
cmap_name = 'my_rainbow'
rainbow   = LinearSegmentedColormap.from_list(cmap_name,colors,N=n_bins)

color = rainbow

labels = [r"$\log \left[ \widetilde{\rho}_{\mathrm{PS}} \, / \, " + 
          r"\widetilde{\rho} \right]$",
          r"$\log \left[ \widetilde{\rho}_{\mathrm{Trujillo+02}} \, / \, " + \
          r"\widetilde{\rho} \right]$",
          r"$\log \left[ \widetilde{\rho}_{\mathrm{SP}} \, / \, " + \
          r"\widetilde{\rho} \right]$",
          r"$\log \left[ \widetilde{\rho}_{\mathrm{LGM}} \, / \, " + \
          r"\widetilde{\rho} \right]$",
          r"$\log \left[ \widetilde{\rho}_{\mathrm{EV}} \, / \, " + \
          r"\widetilde{\rho} \right]$",
          r"$\log \left[ \widetilde{\rho}_{\mathrm{new}} \, / \, " + \
          r"\widetilde{\rho} \right]$",
          r"$\log \left[ \widetilde{M}_{\mathrm{LGM}} \, / \, " + 
          r"\widetilde{M} \right]$",
          r"$\log \left[ \widetilde{M}_{\mathrm{SP}} \, / \, " + \
          r"\widetilde{M} \right]$",
          r"$\log \left[ \widetilde{M}_{\mathrm{new}} \, / \, " + \
          r"\widetilde{M} \right]$"]
func = [np.log10(nu_PS/nu_num),np.log10(nu_Tru/nu_num),np.log10(nu_Sim/nu_num),
        np.log10(nu_LN/nu_num),np.log10(nu_EG/nu_num),LOGnu_numRatVM,
        np.log10(M_LN/M_num),np.log10(M_Sim/M_num),LOGM_numRatVM]
v_min  = -0.1
v_max  = 0.1

fig, axs = plt.subplots(3,3, figsize=(12, 12), facecolor='w', edgecolor='k', \
                        sharey = True)
fig.subplots_adjust(hspace = .4, wspace=.35)
axs = axs.ravel()
for i in range(0,len(axs)) :
    
    axs[i].tick_params(labeltop=False, labelright=False, top = True, right = True, \
                    axis='both', which='major', labelsize=13, direction="in", \
                    length = 8)
    axs[i].tick_params(labeltop=False, labelright=False, top = True, right = True, \
                    axis='both', which='minor', labelsize=13, direction="in", \
                    length = 4)
    if ('new' in labels[i]) :
        axs[i].set_title(labels[i], fontsize = 17, pad = 10, 
                         color=(190/255,0,0))
    else :
        axs[i].set_title(labels[i], fontsize = 17, pad = 10)
    axs[i].set_yscale('log')
    axs[i].set_xscale('log')
    axs[i].set_facecolor('lightgray')
    c = axs[i].pcolor(m, eta, func[i], cmap = color,
                      norm=SymLogNorm(linthresh=0.001, linscale=0.9,
                                              vmin=v_min, vmax=v_max))
    axs[i].set_xticks(np.asarray([0.5,1,2,5,10]))
    axs[i].set_xticklabels(['$0.5$','$1$','$2$','$5$','$10$'])
    cbar = fig.colorbar(c, ax = axs[i])
    cbar.ax.tick_params(labelsize=12.5) 

# Set common labels
fig.text(0.5, 0.07, '$n$', ha='center', va='center', 
         fontsize = 17)
fig.text(0.07, 0.5, '$x = r \, / \, R_{e}$', ha='center', va='center', 
         rotation='vertical', fontsize = 17)


plt.savefig('Grid_Comp.pdf', format = 'pdf', bbox_inches="tight") 
plt.show()

#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
#---------------------------------------------------------------------------
"Residuals RMS"
#---------------------------------------------------------------------------
# Gets the RMS of the ratio between each model and the numerical 
# deprojection over all the grid. Prints first 4 decimal digits

RMS_LN  = model_RMS(nu_LN,nu_num)
RMS_SP  = model_RMS(nu_Sim,nu_num)
RMS_EG  = model_RMS(nu_EG,nu_num)
RMS_PS  = model_RMS(nu_PS,nu_num)
RMS_Tru = model_RMS(nu_Tru,nu_num)
RMS_new = model_RMS(nu_num * 10**LOGnu_numRatVM,nu_num)

print('RMS_PS, density :', round(RMS_PS,4))
print('RMS_LGM, density:',round(RMS_LN,4))
print('RMS_SP, density :', round(RMS_SP,4))
print('RMS_Tru, density:', round(RMS_Tru,4))
print('RMS_EV, density :', round(RMS_EG,4))
print('RMS_new, density:', round(RMS_new,4))

RMS_LNm  = model_RMS(M_LN,M_num)
RMS_SPm  = model_RMS(M_Sim,M_num)
RMS_PSm  = model_RMS(M_PS,M_num)
RMS_newm = model_RMS(M_num * 10**LOGM_numRatVM,M_num)
print('\n\n')
print('RMS_PS, mass :', round(RMS_PSm,4))
print('RMS_LGM, mass:',round(RMS_LNm,4))
print('RMS_SP, mass :', round(RMS_SPm,4))
print('RMS_new, mass:', round(RMS_newm,4))

