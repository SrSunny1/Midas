import cv2
import numpy as np
from openvino.runtime import Core
from threading import Thread
from ultralytics import YOLO
import queue

# Load OpenVINO model
model_xml = "openvino_midas_v21_small_256.xml"  # OpenVINO model
model_bin = "openvino_midas_v21_small_256.bin"  # Model weights
ie = Core()  # OpenVINO runtime core
net = ie.read_model(model=model_xml, weights=model_bin)  # Read model and weights
exec_net = ie.compile_model(model=net, device_name="CPU")  # Compile model on device
input_layer = next(iter(exec_net.inputs))  # Get input layer
output_layer = next(iter(exec_net.outputs))  # Get output layer

# Mobile video stream URL
stream_url = 'http://100.151.23.2:8080/video'
cap = cv2.VideoCapture(stream_url)

if not cap.isOpened():
    print("Cannot open video stream")
    exit()

# Create frame queue
frame_queue = queue.Queue(maxsize=2)  # Maximum 2 frames in queue

# Global variables for depth display
depth_value = None
depth_stats = None

def capture_thread():
    """Thread function to continuously capture frames"""
    while True:
        ret, frame = cap.read()  # Read frame
        if ret:
            if not frame_queue.empty():
                frame_queue.get_nowait()  # Remove old frames if queue is not empty
            frame_queue.put(frame)  # Add new frame to queue


# Mouse callback function to get depth value at clicked position
def mouse_callback(event, x, y, flags, param):
    global depth_value
    if event == cv2.EVENT_LBUTTONDOWN:
        if prediction is not None and x < prediction.shape[1] and y < prediction.shape[0]:
            depth_value = prediction[y, x]
            print(f"Depth value at ({x}, {y}): {depth_value:.2f}")


# Set mouse callback
cv2.namedWindow('Depth Map', cv2.WINDOW_NORMAL)  # Use normal window for resizing
cv2.resizeWindow('Depth Map', 800, 600)  # Set initial window size
cv2.setMouseCallback('Depth Map', mouse_callback)

# Start capture thread
Thread(target=capture_thread, daemon=True).start()  # Daemon thread exits with main thread

frame_counter = 0
prediction = None

while True:
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

    if not frame_queue.empty():
        frame = frame_queue.get()
        frame_counter += 1

        if frame_counter % 2 == 0:  # Process every 2 frames
            # Depth estimation
            img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)  # Convert BGR to RGB
            img = cv2.resize(img, (input_layer.shape[3], input_layer.shape[2]))  # Resize to model input size
            img = np.transpose(img, (2, 0, 1))  # Change to (channels, height, width)
            img = np.expand_dims(img, axis=0)  # Add batch dimension
            img = img.astype(np.float32) / 255.0  # Normalize to [0, 1]

            results = exec_net([img])
            prediction = results[output_layer].squeeze()  # Get depth prediction

            # Calculate depth statistics
            min_depth = np.min(prediction)
            max_depth = np.max(prediction)
            a_depth = np.mean(prediction)

            # Create vertically stacked depth statistics
            depth_stats = [
                f"Min Depth: {min_depth:.2f}",
                f"Max Depth: {max_depth:.2f}",
                f"Avg Depth: {max_depth:.2f}"
            ]

            # Convert to 8-bit for visualization
            depth_map = cv2.normalize(prediction, None, 0, 255, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_8U)

            # Apply color map
            colored_depth = cv2.applyColorMap(depth_map, cv2.COLORMAP_JET)

            # Display depth statistics vertically
            for i, stat in enumerate(depth_stats):
                cv2.putText(colored_depth, stat, (10, 30 + i * 30),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 2)

            # Display selected depth value
            if depth_value is not None:
                cv2.putText(colored_depth, f"Selected Depth: {depth_value:.2f}", (10, 120),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 2)

            # Show original image and depth map
            cv2.imshow('Original Image', frame)
            cv2.imshow('Depth Map', colored_depth)

        # Exit on 'q' key press
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

cap.release()
cv2.destroyAllWindows()