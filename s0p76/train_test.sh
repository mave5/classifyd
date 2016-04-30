#!/usr/bin/env sh

TOOLS=/usr/local/caffe/build/tools
#$TOOLS/caffe train --solver=./net_solver.prototxt -gpu 0 2>&1 | tee caffe-train.log


$TOOLS/caffe train \
--solver=./net_solver.prototxt -gpu 0 --weights=./trainedmodels/VGG16.caffemodel  2>&1 | tee caffe-train.log

#  --solver=./net_solver.prototxt -gpu 0  --snapshot=./trainedmodels/net_iter_30000.solverstate 2>&1 | tee caffe-traindo95.log   
