#!/bin/bash
pip uninstall pycocotools -y
pip install mmpycocotools
MMCV_WITH_OPS=1 pip install git+file:///netscratch/minouei/report/mmcv
pip install git+file:///netscratch/minouei/report/mmdetection
python -u tools/train.py configs/publay/faster_rcnn_s50_fpn_syncbn-backbone+head_mstrain-range_cycle_publay.py  --launcher=slurm
