import cv2
import numpy as np
from openvino.runtime import Core
from threading import Thread
import queue

# 加载 OpenVINO 模型
model_xml = "openvino_midas_v21_small_256.xml"  # openvino模型
model_bin = "openvino_midas_v21_small_256.bin"  # 模型权重
ie = Core()  # openvno运行的核心类
net = ie.read_model(model=model_xml, weights=model_bin)  # 读取模型及其权重
exec_net = ie.compile_model(model=net, device_name="CPU")  # 将模型在指定设备上编译
input_layer = next(iter(exec_net.inputs))  # 读取模型的输入层
output_layer = next(iter(exec_net.outputs))  # 读取模型的输出层

# 手机视频流地址
stream_url = 'http://100.161.173.152:8080/video'
cap = cv2.VideoCapture(stream_url)

if not cap.isOpened():
    print("无法打开视频流")
    exit()

# 创建帧队列
frame_queue = queue.Queue(maxsize = 2) # 队列中最多只有2帧图像

def capture_thread(): # 线程捕获函数
    while True:
        ret, frame = cap.read() # 读取帧
        if ret:
            if not frame_queue.empty():
                frame_queue.get_nowait()  # 队列不为空，移除队列中所有帧（旧帧）
            frame_queue.put(frame)  # 将新读取的帧放入队列中


# 启动捕获线程
Thread(target=capture_thread, daemon=True).start()  # daemon=true，主线程退出时读取线程也会自动终止

frame_counter = 0

while True:
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    if not frame_queue.empty():
        frame = frame_queue.get() # 只取队列中最新帧进行处理

        frame_counter += 1
        if frame_counter % 2 == 0:  # 每 2 帧处理 1 次

            # 深度估计
            img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)  # 将图像从BGR（摄像头读取）格式转化成RGB格式（深度模型需要）
            img = cv2.resize(img, (input_layer.shape[3], input_layer.shape[2]))  # 将帧图像宽高调整为输入层的宽和高
            img = np.transpose(img, (2, 0, 1)) # 将图像通常以 (高度, 宽度, 通道) 的形式存储，但很多模型要求输入的格式为 (通道, 高度, 宽度)
            img = np.expand_dims(img, axis=0)  # 将原来的 (通道, 高度, 宽度) 转换为 (1, 通道, 高度, 宽度) 的形式，其中 1 表示批次大小为 1
            img = img.astype(np.float32) / 255.0  # 转成np.float32,并进行归一化

            results = exec_net([img])
            prediction = results[output_layer]
            prediction = prediction.squeeze()
            depth_map = cv2.normalize(prediction, None, 0, 255, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_8U)

            # 将深度图转换为相应颜色
            orange_colormap = cv2.applyColorMap(depth_map, cv2.COLORMAP_JET)

            # 显示原始图像、目标检测结果和深度图
            cv2.imshow('Original Image', frame)
            cv2.imshow('Depth Map (Orange)', orange_colormap)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

cap.release()
cv2.destroyAllWindows()
