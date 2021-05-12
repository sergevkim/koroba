#!/bin/sh

wget https://www.dropbox.com/s/kojiyv0sxa01k7z/scans.zip
mv scans.zip data
cd data && echo Unzip... && unzip -q scans.zip
echo Unzip is successful!

