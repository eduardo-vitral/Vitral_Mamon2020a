"""
Created on 2019

@author: Eduardo Vitral

"""
###############################################################################
#
# December 2019, Paris
#
# This file prints the results of both linear and cubic interpolations of
# Emsellem & van de Ven 2008 parameters.
#
# Documentation is provided on Vitral & Mamon, 2020a. 
# If you have any further questions please email vitral@iap.fr
#
###############################################################################

import numpy as np
import matplotlib.pyplot as plt
import itertools
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

m_left  = np.log10(0.5) # Initial value of log(n)
m_right = np.log10(10)  # Final value of log(n)
steps_n = 1000          # Number of log(n) bins

#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

def getTru(n,int_method,func) :
    """
    Gets the interpolated parameters from Trujillo et al. 2002 (T+02) density
    fit.
    Considers the spherical case where f = alpha = beta = 1
    func stands for the interpolation function to be used (griddata or splev)
    int_method stands for the interpolation method (e.g. cubic)
    
    """
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
        
    return np.asarray([int_bes,int_p,int_h1,int_h2,int_h3])

def getEG(n,int_method,func) :
    """
    Gets the interpolated parameters from Emsellem & van de Ven 2008 (EV08) 
    density fit.
    func stands for the interpolation function to be used (griddata or splev)
    int_method stands for the interpolation method (e.g. cubic)
    
    """
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
        
    return np.asarray([int_bes,int_p,int_h0,int_h1,int_h2,int_h3])

#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
#---------------------------------------------------------------------------
"Generates data"
#---------------------------------------------------------------------------
# Creates logspaced array of m: Sersic index
m       = np.logspace(m_left,m_right,num = steps_n)

# Gets different cubic interpolation methods
nu_EG_grid = getEG(m,'cubic',True)
nu_EG_splr  = getEG(m,'cubic',False)

# Defines the best fit parameters from Emsellem & van de Ven 2008 (EV08) 
m1    = np.asarray([0.5,0.6,0.7,0.8,0.9,1,1.1,1.2,1.3,1.4,1.5,2,2.5,3,3.5,4,
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
    
params = np.asarray([bes,p,h0,h1,h2,h3])
names  = [r'$\nu$',r'$p$',r'$a_0$',r'$a_1$',r'$10 \times a_2$',
          r'$10 \times a_3$']

#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
#---------------------------------------------------------------------------
"Plotting routine"
#---------------------------------------------------------------------------

fig, axes = plt.subplots(figsize=(10, 5))
colors = ['red', 'b', 'limegreen', 'deepskyblue', 'orange','magenta']
cc = itertools.cycle(colors)
plot_lines = []
plt.tick_params(labeltop=False, labelright=False, top = True, right = True, \
                axis='both', which='major', labelsize=18, direction="in", \
                length = 8)
plt.tick_params(labeltop=False, labelright=False, top = True, right = True, \
                axis='both', which='minor', labelsize=18, direction="in", \
                length = 4)
        
for i in range(0,len(colors)):

    d1 = nu_EG_grid[i]
    
    c = next(cc)
    if (i == 0) :
        axes.semilogx(m1,params[i],color='black', marker='.',
                      markersize=15,linestyle='')
        axes.semilogx(m1,params[i],color='black', linestyle=':')
        axes.semilogx(m,d1, lw=2, color='black')
        
    if (i == 5 or i == 4) :
        axes.semilogx(m,10*d1, lw=2, color=c)
        axes.semilogx(m1,10*params[i],color=c, marker='.', linestyle=':',
                      markersize=15)
    else:
        axes.semilogx(m,d1, lw=2, color=c)
        axes.semilogx(m1,params[i],color=c, marker='.', linestyle=':',
                      markersize=15)

lines = axes.get_lines()
legend2 = plt.legend([lines[i] for i in [3,5,7,9,11,13]],
                     [names[i] for i in range(6)], loc=4, 
                     prop={'size': 18})
legend1 = plt.legend([lines[i] for i in [0,1,2]], ["$\mathrm{Emsellem}" + \
                     " \ \& \ \mathrm{van \ de \ Ven" + \
                     " \ (2008)}$","$\mathrm{Linear" + \
                     "\ interpolation}$",
                     "$\mathrm{Cubic \ " + \
                     "spline \ interpolation}$",
                     "$\mathrm{Linear \ interpolation}$"], loc=1,
                     prop={'size': 18})
axes.add_artist(legend1)
axes.add_artist(legend2)   
plt.xlabel(r'$n$', fontsize = 18)
plt.xticks([m1[3*i] for i in range(int(len(m1)/3))],
           [str('$'+str(m1[3*i])+'$') for i in range(int(len(m1)/3))])

plt.xlim([m[0],m[len(m)-1]])
plt.grid()
plt.minorticks_on()
plt.savefig('SplineTest.pdf', format = 'pdf', bbox_inches="tight") 
plt.show()
