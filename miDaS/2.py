import cv2
import torch
import torchvision.transforms as transforms
from midas.model_loader import load_model

# 模型参数
model_type = "MiDaS_small"
model_path = "weights/midas_small.pt"
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 加载模型
model, transform, net_w, net_h = load_model(device, model_path, model_type, optimize=False, height=None, square=False)

# 打开摄像头
cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # 预处理图像
    img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    input_batch = transform(img).to(device).unsqueeze(0)

    # 进行深度估计
    with torch.no_grad():
        prediction = model(input_batch)
        prediction = torch.nn.functional.interpolate(
            prediction.unsqueeze(1),
            size=img.shape[:2],
            mode='bicubic',
            align_corners=False
        ).squeeze().cpu().numpy()

    # 归一化深度图
    depth_map = cv2.normalize(prediction, None, 0, 255, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_8U)

    # 显示原始图像和深度图
    cv2.imshow('Original Image', frame)
    cv2.imshow('Depth Map', depth_map)

    # 按 'q' 键退出循环
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# 释放摄像头并关闭窗口
cap.release()
cv2.destroyAllWindows()
