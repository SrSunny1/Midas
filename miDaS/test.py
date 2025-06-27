import sys
import os
import cv2
import torch
from torchvision.transforms import Compose, Normalize, ToTensor
from midas.midas_net import MidasNet  # 确保这是正确的模型导入路径

# 获取当前文件的目录
current_dir = os.path.dirname(os.path.abspath(__file__))

# 获取 MiDaS 文件夹的路径
midas_dir = os.path.join(current_dir, 'MiDaS')
print("MiDaS directory:", midas_dir)

# 将 MiDaS 文件夹的路径添加到 sys.path
sys.path.append(os.path.abspath(midas_dir))

# 指定模型权重文件的路径
model_path = "D:/DeepLearning/深度估计/MiDaS/weights/midas_v21_small_256.pt"

# 确保模型文件存在
if not os.path.exists(model_path):
    raise FileNotFoundError(f"Model file not found at {model_path}")

# 初始化模型
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = MidasNet(model_path, non_negative=True).to(device)

# 定义图像预处理
transform = Compose([
    Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ToTensor(),
])

# 打开摄像头
cap = cv2.VideoCapture(0)  # 参数 0 表示使用默认摄像头

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    # 预处理图像
    img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    input_img = transform(img).unsqueeze(0).to(device)

    # 进行深度估计
    with torch.no_grad():
        prediction = model(input_img)
        prediction = torch.nn.functional.interpolate(
            prediction.unsqueeze(1),
            size=img.shape[:2],
            mode='bicubic',
            align_corners=False
        ).squeeze()

    # 归一化深度图
    depth_map = prediction.cpu().numpy()
    depth_map = (depth_map - depth_map.min()) / (depth_map.max() - depth_map.min())

    # 可视化深度图
    cv2.imshow("Depth Map", depth_map)
    cv2.imshow("Original", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()