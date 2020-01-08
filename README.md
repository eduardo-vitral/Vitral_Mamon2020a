# Vitral_Mamon2020a
Main Python codes used in the paper Vitral &amp; Mamon 2020a.
This concerns the deprojection of the Sérsic model (Sérsic 1963; Sersic 1968) onto volume density and mass.
The main files are:

CODES:

sersic_grid_num.py : Calculates the numerical deprojection of the Sérsic model, as well as 
the analytical approximation by Lima Neto, Gerbal & Márquez 1999. Then it fits a polynomial 
to the ratio log(f_LGM/f_numerical) and saves the coefficients.

sersic_grid_comp_rms.py : Compares the new fitted model with other models in the literature

rms_per_n.py : Compares the new fitted model with other models in the literature

slope_comp.py : Compares the LGM approximation with the numerical deprojection.

test_interpolation.py : Plots the interpolation results for the parameters provided in 
Emsellem & van de Ven 2008

create_table_coeff.py : Creates a text file with the table corresponding to the polynomial fits.

einasto_comp.py : Compares the Einasto (1965) profile with the deprojected Sersic profile.

other_models_comp.py : Compares the deprojected Sersic profile with the models from Plummer (1911),
Jaffe (1983), Hernquist (1990) and Navarro et al. (1996).

OTHER FILES:

dens_num.npy, dens_num1000.npy, mass_num.npy, mass_num1000.npy, nu_rh.npy : Numerical calculated values 
for different grids of volume density and mass.

coeff_1.txt, coeff_2.txt : Raw files containing the results of the polynomial fits.

coeff_dens.txt, coeff_mass.txt : Tables containing the results of the polynomial fits.

*** The results and codes here all use the Ciotti & Bertin 1999 approximation for the 
Sérsic parameter, b_n. For the results and codes with the numerical results from equation 2 
from Vitral &amp; Mamon 2020a, please email vitral@iap.fr.

