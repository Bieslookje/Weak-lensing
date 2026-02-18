I compute the matter power spectrum for the standard halo model and the subhalo model in the notebook halo_and_subhalo_model.ipynb. 
The subhalo contribution is computed in a separate notebook: subhalo_param_computation, which makes use of sashimi.c
Using the functions in weak_lensing.py, I compute the angular weak lensing power spectra in GGL_DESY3.ipynb for a given input 3D matter power spectrum.
The pseudo-Cls are computed in PCl.ipynb and stored in a csv file. I also save the namaster workspaces of each galaxy overdensity and shear field as well as their shared covariance fields in fits files.
I perform the final calculation of the SNR values in SNR_calculation.ipynb.
