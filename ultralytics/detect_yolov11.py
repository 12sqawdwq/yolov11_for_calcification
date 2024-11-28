import os
from ultralytics import YOLO
import cv2

def detect_and_draw_yolov11(weights, source, output_dir, conf_thres=0.1):
    # 确保输出目录存在
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # 加载模型
    model = YOLO(weights)
    print(f"Model loaded successfully with device: {model.device}")

    # 获取所有图片路径
    image_paths = [os.path.join(source, f) for f in os.listdir(source) if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
    print(f"Files to process: {image_paths}")

    # 遍历每张图片
    for image_path in image_paths:
        print(f"Processing image: {image_path}")

        # 读取图片
        img = cv2.imread(image_path)
        if img is None:
            print(f"Could not read image: {image_path}, skipping...")
            continue

        # 推理
        results = model.predict(source=img, save=False, conf=conf_thres)

        # 获取检测结果
        for result in results:
            if result.boxes is not None:
                for box in result.boxes:
                    xyxy = box.xyxy[0].cpu().numpy()  # 坐标
                    conf = box.conf[0].item()  # 置信度
                    cls = int(box.cls[0].item())  # 类别

                    # 在图片上绘制检测框
                    cv2.rectangle(img, (int(xyxy[0]), int(xyxy[1])), (int(xyxy[2]), int(xyxy[3])), (0, 255, 0), 2)
                    label = f"{result.names[cls]} {conf:.2f}"
                    cv2.putText(img, label, (int(xyxy[0]), int(xyxy[1]) - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)

        # 保存结果
        save_path = os.path.join(output_dir, os.path.basename(image_path))
        cv2.imwrite(save_path, img)
        print(f"Saved annotated image to {save_path}")

if __name__ == "__main__":
    weights_path = "E:\\TOOL\\yolov11\\runs\\detect\\train6\\weights\\last.pt"
    source_dir = "E:\\TOOL\\datasets\\deteset_for_calcification\\test\\images"
    output_dir = "E:\\TOOL\\yolov11\\runs\\detect\\test_results1"

    detect_and_draw_yolov11(weights=weights_path, source=source_dir, output_dir=output_dir, conf_thres=0.1)
