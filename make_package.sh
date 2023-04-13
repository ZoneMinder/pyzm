#!/bin/bash
echo "rm -rf dist"
rm -fr dist
echo "python setup.py sdist"
python3 setup.py sdist
echo "twine upload dist/*"
twine upload dist/*
