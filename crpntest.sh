#!/bin/bash
pip uninstall pycocotools -y
pip install mmpycocotools
pip install mmcv-full==1.2.5
pip install git+file:///netscratch/minouei/report/mmdetection
python -u tools/test.py configs/publay/crpn_faster_rcnn_x101_freeze_fpn_1x_publay.py /netscratch/minouei/report/work_dirs/crpn_faster_rcnn_rx101_fpn_1x_publay/epoch_3.pth --out crpn-model94-report.pkl --eval bbox --options "classwise=True"  --launcher=slurm
