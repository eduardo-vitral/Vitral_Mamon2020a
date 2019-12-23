"""
Created on 2019
@author: Eduardo Vitral
"""

###############################################################################
#
# December 2019, Paris
#
# This file first computes the numerical values for the deprojection of the 
# Sérsic model (Sérsic 1963; Sersic 1968) onto volume density and mass.
# After, it fits a polynomial to the ratio between the analytical approximation
# of Lima Neto, Gerbal & Márquez 1999 and the numerical calculation. 
#
# Documentation is provided on Vitral & Mamon, 2020a. 
# If you have any further questions please email vitral@iap.fr
#
###############################################################################

import numpy as np
from scipy.integrate import quad
from math import log10, floor
from scipy.special import gamma, gammainc
import time
import math
import glob, os

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

# lin_method is 'True' if you want to compute and save the coefficients
# of the polynomial fit
lin_method = True

order_pol = 10 # polynomial order to fit

#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
#---------------------------------------------------------------------------
"General functions"
#---------------------------------------------------------------------------
    
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
"Numerical calculation of density and mass grids"
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

def integrandM1(x,n) :
    """
    first integrand of the mass (c.f. Vitral & Mamon 2020a)
    
    """
    integral = np.exp(- b(n) * x**(1/n)) * (x**(1/n + 1))
    return integral 
  
def integrandM2(x,s,n) :
    """
    (second+third) integrands of the mass (c.f. Vitral & Mamon 2020a)   
    
    """
    integral = np.exp(- b(n) * x**(1/n)) * (x**(1/n - 1)) * \
                   (np.arcsin(s/x) * x**2 - s*np.sqrt(x**2 - s**2))                
    return integral 

def getM(x,n) :
    """
    constructs the grid of mass values
    
    """
    # Checks if x = r/R_e is an array
    if (np.isscalar(x)) :
        x = np.asarray([x])
    # Checks if n is an array
    if (np.isscalar(n)) :
        n = np.asarray([n])
    M = np.zeros((len(x),len(n))) 
    for i in range(0,len(x)) :
        for j in range(0,len(n)) :
            # computes X-critical (c.f. Vitral & Mamon 2020a)
            x_crit = (9*np.log(10)/b(n[j]))**n[j]
            fact   =  b(n[j])**(2*n[j]+1) / \
                    (gamma(2*n[j]) * n[j]**2)
            if (x_crit <= x[i]) :
                integral1 = 0.5 * quad(integrandM1, 0, x_crit, args=(n[j]),
                                       epsrel=1e-4,epsabs=0,limit=1000)[0] + \
                            0.5 * quad(integrandM1, x_crit, x[i], args=(n[j]),
                                       epsrel=1e-4,epsabs=0,limit=1000)[0]   
          
                integral2 = (1/np.pi) * quad(integrandM2, x[i], np.inf, 
                            args=(x[i],
                            n[j]), epsrel=1e-4,epsabs=0,limit=1000)[0]
    
                M[i,j] = fact * (integral1 + integral2)
            else :    
                integral1 = 0.5 * quad(integrandM1, 0, x[i], args=(n[j]),
                              epsrel=1e-4,epsabs=0,limit=1000)[0]
                
                integral2 = (1/np.pi) * quad(integrandM2, x[i], x_crit, 
                            args=(x[i], n[j]), epsrel=1e-4,epsabs=0,
                            limit=1000)[0] + \
                            (1/np.pi) * quad(integrandM2, x_crit, np.inf, 
                            args=(x[i], n[j]), epsrel=1e-4,epsabs=0,
                            limit=1000)[0]
                
                M[i,j] = fact * (integral1 + integral2)

    return M

#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
#---------------------------------------------------------------------------
"Cunstruction of density and mass grids for known models"
#---------------------------------------------------------------------------

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

#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
#---------------------------------------------------------------------------
"Concerns the polynomial fit"
#---------------------------------------------------------------------------

def getP(x,n,n_params,params) :
    """
    Creates a table of coefficients for the polynomial fit of the grid
    
    """
    P = np.zeros((len(x),len(n))) 
    order = int((-3 + np.sqrt(1 + 8*n_params))/2)
  
    for i in range(0,len(x)) :
        for j in range(0,len(n)) :
            for k in range(0,order+1) :
                for l in range(0,k+1) :
                    
                    P[i,j] += params[int(k*(k+1)/2) + l] * \
                            (np.log10(n[j]))**(k-l) * \
                              (np.log10(x[i]))**l
    return P

def matrix_mult(x,n,order,ratio) :
    """
    Creates the arrays to be given to the function 
    numpy.linalg.lstsq(A,y), such as it is described on the examples 
    section of:
    docs.scipy.org/doc/numpy/reference/generated/numpy.linalg.lstsq.html
    
    """
    A = list()
    y = list()
    
    for i in range(0,len(x)) :
    
        for j in range(0,len(n)) :
            
            if (math.isnan(np.log10(ratio[i,j])) or \
                math.isinf(np.log10(ratio[i,j]))) :
                pass
            else :
                combin = np.zeros(int((order+1)*(order+2)/2))
                
                for k in range(0,order+1) :
                    
                    for l in range(0,k+1) :
                        
                        combin[len(combin) - 1 - int(k*(k+1)/2) - l] = \
                              (np.log10(n[j]))**int(k-l) * \
                              (np.log10(x[i]))**int(l)
                        if (math.isnan(combin[len(combin)-1 - \
                                              int(k*(k+1)/2) - l])) :
                            print('error here!', n[j],x[i],
                                  combin[len(combin)-1 - int(k*(k+1)/2) - l])
                        
                A.append(combin)
                y.append(np.log10(ratio[i,j]))
            
    A = np.asarray(A)
    y = np.asarray(y)
    return A, y

#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
#---------------------------------------------------------------------------
"RMS functions"
#---------------------------------------------------------------------------

def RMS(nu,nuM) :
    """
    Calculates the root-mean-square (rms) of nu and nuM
    
    """
    rms = np.sqrt(np.mean((nu-nuM)**2))
    return rms

def model_P1(params) :
    """
    Gets the root-mean-square (rms) of the fitted ratio for density
    
    """
    nu_m = getP(eta,m,len(params),params)
    
    not_zero1 = np.where(np.logical_not(np.isnan(np.log10(nu_numRat))))
    not_zero2 = np.where(np.logical_not(np.isinf(np.log10(nu_numRat))))
    not_zero  = intersect2d(not_zero1,not_zero2)
    rms = RMS(np.log10(nu_numRat[not_zero]), nu_m[not_zero])

    return rms

def model_P2(params) :
    """
    Gets the root-mean-square (rms) of the fitted ratio for mass
    
    """
    M_m = getP(eta,m,len(params),params)
    
    not_zero1 = np.where(np.logical_not(np.isnan(np.log10(Mass_Rat))))
    not_zero2 = np.where(np.logical_not(np.isinf(np.log10(Mass_Rat))))
    not_zero  = intersect2d(not_zero1,not_zero2)
    rms = RMS(np.log10(Mass_Rat[not_zero]), M_m[not_zero])

    return rms

#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
#---------------------------------------------------------------------------
"main"
#---------------------------------------------------------------------------

# Creates logspaced array of r/R_e
eta    = np.logspace(R_left,R_right,num = steps_R)
# Sam thing for m: Sersic index
m      = np.logspace(m_left,m_right,num = steps_n)

t1_start = time.perf_counter() # Starts a time counter

nu_num = getNU(eta,m) # Gets the grid of density values

nu_LN  = getNUm(eta,m,'LN') # Gets the grid of density values for LGM99
nu_numRat = nu_LN/nu_num # Ratio between LGM99 and numerical density grids

Mass_num = getM(eta,m) # Gets the grid of mass values

Mass_LN  = getMm(eta,m,'LN') # Gets the grid of mass values for LGM99
Mass_Rat = Mass_LN/Mass_num # Ratio between LGM99 and numerical mass grids

# Prints the time taken for calculating the integrals
print("The code took ", (time.perf_counter() - t1_start)/60, 
      " minutes to compute the integrals")

# Saves the results from the numerical calculations to be used after on
np.save('mass_num', Mass_num) 
np.save('dens_num', nu_num)


#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
#---------------------------------------------------------------------------
"Fitting procedure"
#---------------------------------------------------------------------------

model = [model_P1, model_P2]
func  = [nu_numRat, Mass_Rat]

# Here, we fit the best polynomial to both mass and density profiles ratio
# f_LGM / f_numerical
if (lin_method == True) :
    # WARNING: Removes previous .txt files to avoid overwriting
    for file in glob.glob("*.txt"):
        if('coeff_1.txt' == file or 'coeff_2.txt' == file) :
            os.remove(file)

    for k in range(0,len(model)) :
        
        t1_start = time.perf_counter()
        A, y  = matrix_mult(eta,m,order_pol,func[k])
        coeff = np.linalg.lstsq(A,y,rcond=None)[0]
        # Prints the information on rms and time to find the best fit
        print('Order ', str(order_pol), ': rms = ', model[k](coeff[::-1]), 
              ' / time = ', RoundIt((time.perf_counter() - t1_start)/60),
              ' minutes')
        # Saves the coefficients
        with open('coeff_' + str(k+1) + '.txt', 'a') as f:
            print('\nOrder ', str(order_pol), ': took ', 
              (time.perf_counter() - t1_start)/60, ' minutes', file=f)
            print('rms: ', model[k](coeff[::-1]), '\n\n', file=f)
            for j in range(0,len(coeff)) :
                print(coeff[j], file=f)
                
        print('\n\n')   