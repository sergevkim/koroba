#!/bin/sh

cd koroba/utils/ops && python setup.py install && cd ../../..
pip install -e .
rm -rf koroba.egg-info

