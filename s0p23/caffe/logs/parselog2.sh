#!/usr/bin/env sh

cafferoot=~/code/caffe

python $cafferoot/tools/extra/parse_log.py caffelog2 .
mv caffelog2.test caffelogtest.txt
mv caffelog2.train caffelogtrain.txt
