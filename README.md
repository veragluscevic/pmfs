pmfs
====
Data and code accompanying [Gluscevic et al. (2016a)](http://arxiv.org/abs/1604.06327). The code in this repository can be used to:

*  Calculate the effect of cosmological magnetic fields on 21-cm brightness-temperature fluctuations
*  Forecast sensitivities of 21-cm tomographic surveys to detecting cosmological magnetic fields, using the new method of this paper series


Data
----

Figures in the [paper](https://github.com/veragluscevic/pmfs/blob/master/paper/detectability.pdf) use outputs of ``21CMFAST`` code that are in ``code/inputs`` and outputs of the ``CAMB`` code in ``code/matter_power``. Numerical results represented in our Figures are in ``code/results``.


Dependencies
------------
This code will only work for python 2. For calculations, you will need:

* basic python scientific packages contained within the `Anaconda python distribution <http://continuum.io/downloads>`_ (``numpy``, ``scipy``, ``matplotlib``)
* ``cosmolopy``
* ``numba``

For producing Fig. 2, you will also need ``Healpix`` and ``Healpy`` (see [here](https://github.com/healpy/healpy>)).
 
 
Usage
----------
There are three main scripts you may want to use and can import from python command line: 

* ``fisher.py``: Enables computation of the detection threshold for a given 21-cm survey (note: only implemented for an array of dipoles in a compact-grid configuration). The main routines are:

    - ``rand_integrator``: It takes on input ``zmin``, ``zmax``, ``Omega_survey`` (sr), and ``Delta_L`` (km) of the survey. For default values, it will give you the first data point of Fig. 5 for the uniform-field case. If you set ``mode='xi'``, it will give you the first data point of Fig. 4 for fiducial model.
    - ``calc_SNR``: It takes the same survey parameters as ``rand_integrator``. If you use default parameters, you will get the first data point of Fig. 5 for the SI case.


* ``plots_pmfs.py``: If you are looking to use this module to reproduce Figures in the [paper](https://github.com/veragluscevic/pmfs/blob/master/paper/detectability.pdf), you will need this routine and you can read ``code/plotting_commands_for_paper.txt`` on how to use it to do that. Note that the plotting routine that produces Figs. 4 and 5 (the main result of the [paper](https://github.com/veragluscevic/pmfs/blob/master/paper/detectability.pdf)) is ``grid_DeltaL``, and it requires that you first compute sensitivities for a range of maximum baselines, using appropriate routines of ``fisher``. The format of the inputs to the plotting routine is as contained in ``code/results/midFSTAR``, ``code/results/loFSTAR``, and ``code/results/hiFSTAR``. You can also just run the plotting routine for the our results in ``results`` to get the figures.


* ``pmfs_transfer.py``: This module contains a basic implementation of the brightness-temperature calculation. The main routines are:

      - ``calc_Tb``: implementation of Eq. (1) (computation of the brightness-temperature fluctuation in presence of a magnetic field) 
      - ``calc_G``: implementation of Eq. (24) (the corresponding transfer function)
      
  Both routines take on input:
       - values of $x_{\alpha,(2)}$, $x_{c,(2)}$, and $x_B$ (all are functions of redshift; see the [paper](https://github.com/veragluscevic/pmfs/blob/master/paper/detectability.pdf) for definitions),
       - coordinates of the wavevector $\vec k$ and the line of sight $\widehat n$ (both in the frame where the magnetic field is along the z axis; angle coordinates are in radians),
  while the second routine also takes spin and CMB temperatures at a given redshift (in Kelvin).

This should get you quickly started if you wish to reproduce our results; for more details, see documentation within the source code and in the [paper](https://github.com/veragluscevic/pmfs/blob/master/paper/detectability.pdf).

Additional notes
----------------

* If you wish to parallelize your sensitivity calculation, check out ``code/cluster_commands_for_paper.txt`` and ``code/cluster_runner.py`` for inspiration. 

* Code for lensing-noise calculation of Appendix B is in ``code/lensing-code``; see ``README`` there for more information.

Attribution
-----------

If you use this code, please cite Gluscevic et al. (2016a) and Venumadhav et al (2014).



