Installation and Dependencies
-----------------------------
The current release of spaceKLIP is not stable enough for a straightforward install with conda / pip. At this stage
it is recommended that you clone the git repository directory for installation:

::

	git clone https://github.com/kammerje/spaceKLIP.git

If you would like a specific branch:

::

	git clone https://github.com/kammerje/spaceKLIP.git@branch

From here, it is **highly** recommended that you create a unique anaconda environment to hold all of the spaceKLIP
dependencies. spaceKLIP is not currently compatible with python 3.12

::

	conda create -n spaceklip python=3.11
	conda activate spaceklip

With the anaconda environment created, move to the cloned directory and install most of the dependencies:

::

	cd where/you/installed/the/git/repo
	pip install -r requirements.txt
	pip install -e .

Finally, and very importantly, you will need to download reference files to support the functioning of the :code:`webbpsf` and :code:`webbpsf_ext`. Instructions to do this can be found at the respective package websites (`WebbPSF <https://webbpsf.readthedocs.io/en/latest/installation.html#installing-the-required-data-files>`_, `webbpsf_ext <https://github.com/JarronL/webbpsf_ext>`_). Ensure that if you edit your .bashrc file that you reopen and close your terminal to fully apply the changes (:code:`source ~/.bashrc` or :code:`source ~/.zshrc` may also work)


SpaceKLIP also makes use of the JWST Calibration Reference Data System (CRDS) and you will need to set the corresponding environment variables. Follow the instructions here for bash or zsh: https://jwst-crds.stsci.edu/docs/cmdline_bestrefs/. Note that you do not have to install AstroConda, just set the environment variables (making sure that the CRDS path you set actually exists, i.e., you may need to create the directory).