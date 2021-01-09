#!/bin/bash
pip uninstall pycocotools -y
pip install mmpycocotools
pip install mmcv-full==1.2.5
pip install git+file:///netscratch/minouei/report/mmdetection
python -u tools/train.py configs/publay/crpn_faster_rcnn_r50_caffe_fpn_dconv_1x_publay.py  --launcher=slurm
