.PHONY: help pypi pypi-test docs coverage test clean

docs:
	rm -rf docs/api
	rm -f docs/source/spaceKLIP.companion.rst
	rm -f docs/source/spaceKLIP.contrast.rst
	rm -f docs/source/spaceKLIP.engine.rst
	rm -f docs/source/spaceKLIP.imgprocess.rst
	rm -f docs/source/spaceKLIP.io.rst
	rm -f docs/source/spaceKLIP.plotting.rst
	rm -f docs/source/spaceKLIP.rampfit.rst
	rm -f docs/source/spaceKLIP.rst
	rm -f docs/source/spaceKLIP.subtraction.rst
	rm -f docs/source/spaceKLIP.utils.rst
	sphinx-apidoc -o docs/source spaceKLIP
	cd docs/source/
	$(MAKE) -C docs clean
	$(MAKE) -C docs html
