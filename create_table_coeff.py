"""
Created on 2019
@author: Eduardo Vitral
"""
###############################################################################
#
# December 2019, Paris
#
# This file saves a text file with the coefficients of the polynomial fit.
# Columns vary with the exponent of log(n) and rows vary with the exponent 
# of log(r/R_e).
#
# Documentation is provided on Vitral & Mamon, 2020a. 
# If you have any further questions please email vitral@iap.fr
#
###############################################################################

import numpy as np
import matplotlib.pyplot as plt
from math import log10, floor, ceil

plt.rcParams['font.family'] = 'serif'
plt.rcParams['font.serif'] = ['Computer Modern Roman'] \
    + plt.rcParams['font.serif']

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

def PowerTen (Number):
    """
    Returns a number in power ten representation in form of an array, 
    with the power of ten in the second case and the number multiplied 
    by the tenth power in the first case, up to its first three significant
    digits
    
    """
    TenPower = np.zeros(2)
    if(Number != 0) :
        TenPower[1] = floor(log10(np.abs(Number)))
    else : TenPower[1] = 0
    TenPower[0] = round(Number/10**TenPower[1],3)
    return TenPower

#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
#---------------------------------------------------------------------------
"Formatting functions"
#---------------------------------------------------------------------------

def PolynomialM(params) :
    """
    Fills a matrix with the coefficients of the polynomial fit for each
    power combination of log(r/R_e) and log(n)
    
    """
    n_params = len(params)
    order    = int((-3 + np.sqrt(1 + 8*n_params))/2) + 1
    matrix   = [[0]*order for i in range(order)]

    for k in range(1,n_params+1) :
        p           = ceil((-3 + np.sqrt(1 + 8*k))/2)
        x_exp       = int(k - 1 - p*(p+1)/2)
        n_exp       = int(1 - k + p*(p+3)/2)
        coeff_value = PowerTen(params[n_params -(k-1) -1]) 

        matrix[n_exp][x_exp] = float("{: .3f}".format(-coeff_value[0]) + \
                               'e' + str(int(coeff_value[1])))
    
    return matrix

def writeInFile (data,Name) :
    """
    Writes coefficients of the polynomial fit in a text file
    
    """
    table = open(Name, "w")
    
    table.write('# Table of coefficients. ' + \
                      '\n# rows   : increase according to the ' + \
                      'exponent of log (r/R_e)' + \
                      '\n# columns: increase according to the' + \
                      ' exponent of log (n)' + \
                      '\n')
    for i in range(0,np.shape(data)[1]) :
        # write line to output file
        string = '{: 11.3e} {: 11.3e} {: 11.3e} {: 11.3e} {: 11.3e} ' + \
                 '{: 11.3e} {: 11.3e} {: 11.3e} {: 11.3e} {: 11.3e} {: 11.3e}'
        line = string.format(data[0][i], data[1][i], \
                data[2][i], data[3][i], data[4][i], data[5][i], data[6][i], \
                data[7][i],data[8][i],data[9][i],data[10][i])
      
        table.write(line)
        table.write("\n")
    
    table.close()
    
#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
#---------------------------------------------------------------------------
"main"
#---------------------------------------------------------------------------

# Gets the polynomial fit coefficients
coeff_nu = getCoeff('coeff_1.txt')
coeff_M  = getCoeff('coeff_2.txt')

# Gives the coefficients in a matrix-shape array
coeff_nu = PolynomialM(coeff_nu)
coeff_M  = PolynomialM(coeff_M)

# Writes the coefficients in a file
writeInFile(coeff_nu,'coeff_dens.txt')
writeInFile(coeff_M,'coeff_mass.txt')