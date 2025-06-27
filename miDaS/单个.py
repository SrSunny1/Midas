import cv2
import numpy as np
from openvino.runtime import Core
import sys

# 加载OpenVINO模型
model_xml = "openvino_midas_v21_small_256.xml"
model_bin = "openvino_midas_v21_small_256.bin"

ie = Core()
net = ie.read_model(model=model_xml, weights=model_bin)
exec_net = ie.compile_model(model=net)
input_layer = next(iter(exec_net.inputs))
output_layer = next(iter(exec_net.outputs))

# 直接指定图片路径，无需命令行参数
image_path = "1.jpg"
frame = cv2.imread(image_path)

if frame is None:
    print(f"无法读取图片: {image_path}")
    sys.exit(1)

# 深度估计
img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
img = cv2.resize(img, (input_layer.shape[3], input_layer.shape[2]))
img = np.transpose(img, (2, 0, 1))
img = np.expand_dims(img, axis=0)
img = img.astype(np.float32) / 255.0

# 深度预测
results = exec_net([img])
prediction = results[output_layer].squeeze()

# 转换为8位用于可视化
depth_map = cv2.normalize(prediction, None, 0, 255, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_8U)

# 应用颜色映射
colored_depth = cv2.applyColorMap(depth_map, cv2.COLORMAP_JET)

# 调整深度图尺寸为原始图像大小
colored_depth = cv2.resize(colored_depth, (frame.shape[1]//2, frame.shape[0]//2))

# 显示深度图
cv2.imshow('Depth Map', colored_depth)
print("深度图已生成，按任意键退出")

cv2.waitKey(0)
cv2.destroyAllWindows()
