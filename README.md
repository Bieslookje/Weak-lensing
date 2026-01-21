I compute the matter power spectrum for the standard halo model and the subhalo model in the notebook halo_and_subhalo_model.ipynb. 
For the subhalo parameters, I use the functions 'mass_fraction', 'mass_function', 'density_parameters' from the subhalo_observables class in sashimi.c. 
Using the functions in DES_CDM.py, I compute the angular weak lensing power spectra in Cl_calculation.ipynb for a giveninput 3D matter power spectrum, and save the results as numpy arrays. 
The pseudo-Cls are computed in PCl.ipynb and stored in a csv file. I also save the namaster workspaces of each galaxy overdensity and shear field as well as their shared covariance fields in fits files.
I perform the final calculation of the SNR values in SNR_calculation.ipynb.
