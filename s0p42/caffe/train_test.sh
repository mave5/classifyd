#!/usr/bin/env sh

TOOLS=/usr/local/caffeapril2016/build/tools

$TOOLS/caffe train \
 --solver=./net_solver.prototxt -gpu 1 --weights=./trainedmodels/ResNet-50-model.caffemodel  2>&1 | tee caffelog.log 

#  --solver=./net_solver.prototxt -gpu 1  --snapshot=./trainedmodels/net_iter_10000.solverstate 2>&1 | tee #caffelog2.log  



