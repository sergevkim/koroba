#!/bin/sh

pip install -e .
rm -rf koroba.egg-info
cd koroba/utils/ops && python setup.py install

