#!/bin/bash
pip uninstall pycocotools -y
pip install mmpycocotools
pip install mmcv-full==1.2.5
pip install git+file:///netscratch/minouei/report/mmdetection
python -u tools/train.py configs/publay/dh_faster_rcnn_x101_freeze_fpn_1x_publay.py  --launcher=slurm
