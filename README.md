pmfs
====
Code accompanying Gluscevic et al. (2016a) that can be used to:

*  Calculate the effect of cosmological magnetic fields on 21-cm brightness-temperature fluctuations
*  Forecast sensitivities of 21-cm tomographic surveys to detecting cosmological magnetic fields using the method of Venumadhav et al. (2014)

Dependencies
------------
This code will only work for python 2.

For calculations, you will need:

* basic python scientific packages contained within the `Anaconda python distribution <http://continuum.io/downloads>`_ (``numpy``, ``scipy``, ``matplotlib``)
* ``cosmolopy``
* ``numba``

For plotting (just for figures like Fig. 2 in the paper) you may also need ``Healpix`` and ``Healpy`` (see `here <https://github.com/healpy/healpy>`_).

Data
----

Figures in Gluscevic et al. (2016a) use outputs of ``21CMFAST`` code that are contained in ``code/inputs`` and outputs of the ``CAMB`` code contained in ``code/matter_power``. Numerical results represented in our Figures is contained in ``code/results``.

Usage
------
There are three main scripts you may want to use and can import from python command line: 

* ``fisher.py``: Enables computation of the detection threshold for a given 21-cm survey (note: only implemented for an array of dipoles in a compact-grid configuration). The main routines are:

    - ``rand_integrator``: It takes on input ``zmin``, ``zmax``, ``Omega_survey`` (sr), and ``Delta_L`` (km) of the survey. For default values, it will give you the first data point of Fig. 5 for the uniform-field case. If you set ``mode='xi'``, it will give you the first data point of Fig. 4 for fiducial model.
    - ``calc_SNR``: It takes the same survey parameters as ``rand_integrator``. If you use default parameters, you will get the first data point of Fig. 5 for the SI case.


* ``plots_pmfs.py``: If you are looking to use this module to reproduce figures of the paper, read ``code/plotting_commands_for_paper.txt``. Note that the plotting routine that produces Figs. 4 and 5 is ``grid_DeltaL``, and it requires that you first compute sensitivities for a range of maximum baselines, using appropriate routines of ``fisher``. The format of this input is as contained in ``code/results/midFSTAR``, ``code/results/loFSTAR``, and ``code/results/hiFSTAR``. You can also just run the plotting routines for the our results in ``results`` to get the figures.


* ``pmfs_transfer.py``: Basic implementation of the brightness-temperature calculation.

      - for implementation of Eq. (1) (computation of the brightness-temperature fluctuation in presence of a magnetic field) use ``calc_Tb`` 
      - for Eq. (24) (the corresponding transfer function) use ``calc_G`` 
  Both routines take on input:
       - values of $x_{\alpha,(2)}$, $x_{c,(2)}$, and $x_B$ (see paper for definitions) which are all functions of redshift,
       - coordinates of the wavevector $\vec k$ and the line of sight $\widehat n$ (both in the frame where the magnetic field is along the z axis; angle coordinates are in radians),
  while the second routine also takes spin and CMB temperatures at a given redshift (in Kelvin).



Additional notes
----------------

* If you wish to parallelize your sensitivity calculation, check out ``code/cluster_commands_for_paper.txt`` and ``code/cluster_runner.py``. You won't be able to directly use these, but they may serve as inspiration.

* Code for lensing-noise calculation of Appendix B is in ``code/lensing-code``; see ``README`` there for more information.


