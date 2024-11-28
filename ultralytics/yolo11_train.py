import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'
from ultralytics import YOLO
if __name__ == '__main__':
    model = YOLO('ultralytics/cfg/models/11/yolo11n.yaml')  # 从YAML建立一个新模型
    model.load('yolo11n.pt')
    # # 训练模型
    results = model.train(data='E:\TOOL\yolov11\\ultralytics\data.yml', epochs=200, imgsz=640, device=0, optimizer='SGD', workers=8, batch=8, amp=False)