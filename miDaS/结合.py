import cv2
import numpy as np
from openvino.runtime import Core
from threading import Thread
from ultralytics import YOLO
import queue

# 加载OpenVINO模型
model_xml = "openvino_midas_v21_small_256.xml"  # OpenVINO模型文件
model_bin = "openvino_midas_v21_small_256.bin"  # 模型权重文件

yolo_model = YOLO("yolov8n.pt")

person_id = 0
ie = Core()  # OpenVINO运行时核心
net = ie.read_model(model=model_xml, weights=model_bin)  # 读取模型和权重
exec_net = ie.compile_model(model=net, device_name="CPU")  # 在设备上编译模型
input_layer = next(iter(exec_net.inputs))  # 获取输入层
output_layer = next(iter(exec_net.outputs))  # 获取输出层

# 移动视频流URL
stream_url = 'http://100.178.114.192:8080/video'
cap = cv2.VideoCapture(stream_url)


# 调整输入图象大小（与输出图象无关）
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

if not cap.isOpened():
    print("无法打开视频流")
    exit()

# 创建帧队列
frame_queue = queue.Queue(maxsize=2)  # 队列中最多2帧

# 深度显示的全局变量
depth_value = None
depth_stats = None

def capture_thread():
    """线程函数，持续捕获帧"""
    while True:
        ret, frame = cap.read()  # 读取帧
        if ret:
            if not frame_queue.empty():
                frame_queue.get_nowait()  # 如果队列不为空，移除旧帧
            frame_queue.put(frame)  # 将新帧添加到队列


# 鼠标回调函数，获取点击位置的深度值
def mouse_callback(event, x, y, flags, param):
    global depth_value
    if event == cv2.EVENT_LBUTTONDOWN:
        if prediction is not None and x < prediction.shape[1] and y < prediction.shape[0]:
            depth_value = prediction[y, x]
            print(f"坐标({x}, {y})的深度值: {depth_value:.2f}")


# 设置鼠标回调
cv2.namedWindow('Depth Map', cv2.WINDOW_NORMAL)  # 使用普通窗口以便调整大小
cv2.resizeWindow('Depth Map', 640, 669)  # 设置初始窗口大小
cv2.setMouseCallback('Depth Map', mouse_callback)

# 启动捕获线程
Thread(target=capture_thread, daemon=True).start()  # 守护线程随主线程退出

frame_counter = 0
prediction = None

while True:

    if not frame_queue.empty():
        frame = frame_queue.get()
        frame_counter += 1

        if frame_counter % 2 == 0:  # 每2帧处理一次
            # 深度估计
            img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)  # BGR转RGB
            img = cv2.resize(img, (input_layer.shape[3], input_layer.shape[2]))  # 调整为模型输入大小
            img = np.transpose(img, (2, 0, 1))  # 改为(通道, 高度, 宽度)
            img = np.expand_dims(img, axis=0)  # 添加批次维度
            img = img.astype(np.float32) / 255.0  # 归一化到[0, 1]

            rs = yolo_model(frame)

            r = rs[0].plot()

            results = exec_net([img])
            prediction = results[output_layer].squeeze()  # 获取深度预测结果

            # 计算深度统计信息
            min_depth = np.min(prediction)
            max_depth = np.max(prediction)
            a_depth = np.mean(prediction)

            # 创建垂直排列的深度统计信息
            depth_stats = [
                f"Min_Depth: {min_depth:.2f}",
                f"Max_Depth: {max_depth:.2f}",
                f"Avg_Depth: {a_depth:.2f}"
            ]

            # 转换为8位用于可视化
            depth_map = cv2.normalize(prediction, None, 0, 255, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_8U)

            # 应用颜色映射
            colored_depth = cv2.applyColorMap(depth_map, cv2.COLORMAP_JET)

            # 垂直显示深度统计信息
            for i, stat in enumerate(depth_stats):
                cv2.putText(colored_depth, stat, (10, 30 + i * 30),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)

            # 显示选中的深度值
            if depth_value is not None:
                cv2.putText(colored_depth, f"check_area: {depth_value:.2f}", (10, 120),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 2, 255), 1)

            # 示例：保持输入高分辨率，仅调整显示大小
            resized_display = cv2.resize(r, (640, 690))  # 调整显示尺寸但不影响模型输入

            # 显示原始图像和深度图
            cv2.imshow('Original Image', resized_display)
            cv2.imshow('Depth Map', colored_depth)

        # 按'q'键退出
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

cap.release()
cv2.destroyAllWindows()