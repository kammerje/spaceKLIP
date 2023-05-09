##############
spaceKLIP 🚀🪐
##############

spaceKLIP is a data reduction pipeline for JWST high contrast imaging. All code is currently under heavy development
and typically expected features may not be available.

Compatible Simulated Data: `Here <https://stsci.box.com/s/cktghuyrwrallb401rw5y5da2g5ije6t>`_
Compatible On Sky Data: `Here <https://stsci.app.box.com/s/awowbrrf7lhhb69xi4qlkliwv7euk6os>`_

Installation Instructions
*************************

The current release of spaceKLIP is not stable enough for a straightforward install with conda / pip. At this stage
it is recommended that you clone the git repository directory for installation:

::

	git clone https://github.com/kammerje/spaceKLIP.git

If you would like a specific branch:

::

	git clone https://github.com/kammerje/spaceKLIP.git@branch

From here, it is **highly** recommended that you create a unique anaconda environment to hold all of the spaceKLIP
dependencies.

::

	conda create -n sklip python=3
	conda activate sklip

With the anaconda environment created, move to the cloned directory and install most of the dependencies:

::

	cd where/you/installed/the/git/repo
	pip install -e .

You will also need to install some other specific packages, namely:

::

	pip install git+https://bitbucket.org/pyKLIP/pyklip.git@jwst
	pip install git+https://github.com/JarronL/webbpsf_ext.git@develop

Finally, and very importantly, you will need to download reference files to support the functioning of
the :code:`webbpsf` and :code:`webbpsf_ext`. Instructions to do this can be found at the respective package websites (`WebbPSF <https://webbpsf.readthedocs.io/en/latest/installation.html#installing-the-required-data-files>`_, `webbpsf_ext <https://github.com/JarronL/webbpsf_ext>`_). Ensure that if you edit your .bashrc file that you reopen and close your terminal to fully apply the changes (:code:`source ~/.bashrc` or :code:`source ~/.zshrc` may also work)

spaceKLIP also makes use of the JWST Calibration Reference Data System (CRDS) and you will need to set some environment variables. Follow the instructions here for bash or zsh: https://jwst-crds.stsci.edu/docs/cmdline_bestrefs/
Note you do not have to install astroconda, just set the environment variables (taking care that the CRDS path you set actually exists, i.e., you may need to create the directory).

If you want to run the tutorial notebook, you will need to install jupyter:

::

	pip install jupyter

If you want to use nested sampling to fit forward models, you will need to install pymultinest:

::

	conda install -c conda-forge pymultinest
