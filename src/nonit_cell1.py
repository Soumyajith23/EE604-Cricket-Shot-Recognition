# 1. pre-download the pose weights to an absolute path (so both GPU workers find them)
if not os.path.exists(POSE):
    from ultralytics import YOLO
    import shutil
    YOLO('yolo11m-pose.pt')                      # downloads into the current dir
    for cand in ('yolo11m-pose.pt', os.path.expanduser('~/yolo11m-pose.pt')):
        if os.path.exists(cand) and os.path.abspath(cand) != POSE:
            shutil.copy(cand, POSE); break
print(os.path.exists(POSE), os.path.getsize(POSE) / 1e6, 'MB')