##############
spaceKLIP 🚀🪐
##############

spaceKLIP is a data reduction pipeline for JWST high contrast imaging. All code is currently under heavy development
and typically expected features may not be available. 

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
	pip install -r requirements.txt

You will also need to install some other specific packages, namely:

::

	pip install git+https://bitbucket.org/pyKLIP/pyklip.git@jwst
	pip install git+https://github.com/JarronL/webbpsf_ext.git

Finally, and very importantly, you will need to download reference files to support the functioning of 
the :code:`webbpsf` and :code:`webbpsf_ext`. Instructions to do this can be found at the respective package websites (`WebbPSF <https://webbpsf.readthedocs.io/en/latest/installation.html#installing-the-required-data-files>`_, `webbpsf_ext <https://github.com/JarronL/webbpsf_ext>`_). Ensure that if you edit your .bashrc file that you reopen and close your terminal to fully apply the changes (:code:`source ~/.bashrc` or :code:`source ~/.zshrc` may also work)