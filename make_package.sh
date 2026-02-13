#!/bin/bash
echo "rm -rf dist"
rm -fr dist
echo "python -m build"
python3 -m build
echo "twine upload dist/*"
twine upload dist/* --verbose
