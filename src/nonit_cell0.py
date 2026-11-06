# 0. setup
!pip install -q ultralytics
import os, subprocess, sys, glob, pickle, time
DS   = '/kaggle/input/datasets/nonitgupta/crickshot10k'
MD   = '/kaggle/input/datasets/nonitgupta/player-type-detection'
BAT  = f'{MD}/Bat_Detection_Model.pt'
SEG  = f'{MD}/Striker_Bat_Segmentation_Model.pt'
POSE = '/kaggle/working/yolo11m-pose.pt'
for p in (DS, BAT, SEG): print(os.path.exists(p), p)