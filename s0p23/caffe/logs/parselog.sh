#!/usr/bin/env sh

cafferoot=~/code/caffe

python $cafferoot/tools/extra/parse_log.py caffelog .
mv caffelog.test caffelogtest.txt
mv caffelog.train caffelogtrain.txt
