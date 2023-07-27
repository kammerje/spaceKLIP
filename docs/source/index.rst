.. This file should at least contain the root `toctree` directive.


Documentation for spaceKLIP
===========================

.. image:: _static/logo.png
   :align: center
   :width: 400px

spaceKLIP is a data reduction pipeline for JWST high contrast imaging, particularly NIRCam and MIRI coronagraphy.
It is built on top of `pyKLIP <https://pyklip.readthedocs.io/en/latest/>`_ and provides functionality and customizations specific to JWST data. 

All code is currently under heavy development
and typically expected features may not be available. 


.. note::
   **Citing spaceKLIP:** 
   If spaceKLIP is useful for your research, 
   please cite 
   `Kammerer et al. 2022, Proc SPIE. <https://ui.adsabs.harvard.edu/abs/2022SPIE12180E..3NK/abstract>`_ and
   `Carter et al. 2023, ApJL <https://ui.adsabs.harvard.edu/abs/2023ApJ...951L..20C/abstract>`_. 


Compatible Simulated Data: `Here <https://stsci.box.com/s/cktghuyrwrallb401rw5y5da2g5ije6t>`_

.. toctree::
   :maxdepth: 2
   :caption: Getting started
   :hidden:

   TLDR
   Installation-and-dependencies
   Config_file

.. toctree::
   :maxdepth: 3
   :caption: Tutorials
   :hidden:

   tutos
   tutorials/Introduction.ipynb
   tutorials/01_quickstart.ipynb

.. toctree::
   :maxdepth: 2
   :caption: About
   :hidden:

   Contact
   Attribution

.. toctree::
   :maxdepth: 2
   :caption: Package content
   :hidden:

   spaceKLIP
   gen_index

