#!/bin/bash
wget https://download.microsoft.com/download/8/7/0/8700516A-AB3D-4850-B4BB-805C515AECE1/FB15K-237.2.zip
unzip FB15K-237.2.zip -d data/FB15K237
mv data/FB15K237/Release/* data/FB15K237/
rm -r data/FB15K237/Release/
rm FB15K-237.2.zip


wget https://data.deepai.org/WN18RR.zip --no-check-certificate
unzip WN18RR.zip  -d data/
mv data/WN18RR/original/* data/WN18RR/
rm -r data/WN18RR/original/
rm WN18RR.zip


wget https://github.com/TimDettmers/ConvE/raw/master/YAGO3-10.tar.gz
mkdir data/YAGO310
tar -xvzf YAGO3-10.tar.gz --directory data/YAGO310
rm YAGO3-10.tar.gz

python data/create_symmetry_list.py
git clone https://github.com/nec-research/KGEval_hierarchy tests/hierarchy/resources
mv tests/hierarchy/resources/resources/* tests/hierarchy/resources/
rm -rf tests/hierarchy/resources/resources/
