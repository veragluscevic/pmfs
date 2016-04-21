pmfs
====
Code accompanying Gluscevic et al. 2016a that enables two main functions:

* Calculating the effect of cosmological magnetic fields on 21-cm brightness-temperature fluctuations
* Forecasting sensitivities of 21-cm tomographic surveys to cosmological magnetic fields

Dependencies
------------
This code will only work for python 2.

For calculations, you will need:

* basic python scientific packages contained within the `Anaconda python distribution <http://continuum.io/downloads>`_ (``numpy``, ``scipy``, ``matplotlib``)
* ``cosmolopy``
* ``numba``

For plotting (just for figures like Fig. 2 in the paper) you may also need ``Healpix`` and ``Healpy``

Usage
------
There are 3 main scripts you may wish to use: ``pmfs_transfer.py``, ``fisher.py``, and ``plots_pmfs.py``.

* Numerical results

To compute brightness-temperature fluctuation from Eq. (1) of the paper, from pyhton command line:

>> import fisher as f
>> f.calc_Tb()

where in particular you need to pass values of x_\alpha, x_c, and xB, and the coordinates of wavevector k and line of sight n, both in the frame where the magnetic field is along the z axis. You can similarly compute the transfer function G using the same script, and calling f.calc_G().

To compute detection threshold for experiments:

>> import fisher as f
>> f.rand_integrator()

where you need to pass zmin, zmax, Omega_survey, and \Delta L of your survey. If you pass values quoted in the paper, this will give you the first data point of Fig. 5 for the uniform-field case. The same function computes data points in Fig. 4, if you sent mode='xi'. For the case of the scale-invariant power spectrum of a stochastic field, you need:

>> f.calc_SNR()

and pass to it, again, parameters of the survey. If you use default parameters, you will get the first data point of Fig. 5 for the SI case.

* Plotting

Details on plotting are described in a separate file, plotting_commands_for_paper.txt. The format of the inputs that those commands requre is the following...


Additional notes
----------------

* If you wish to parallelize your sensitivity calculation...

* Code for lensing-noise calculation of Appendix B is in lensing-code directory; see README there for more information.


