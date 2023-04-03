#import tensorflow as tf
import tensorflow_hub as hub
import cv2
import os
#cpu运行，要想GPU删除下面两行
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID" 
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
import tensorflow as tf
import numpy as np
import mss
import pyautogui
from PIL import Image
from pynput import mouse
import time
# with tf.device('/cpu:0'):
# Load the MoveNet model from TensorFlow Hub
# Download the MoveNet model to a local directory
# Load the MoveNet model from the local directory
model_path = "C:\\Users\\ocean\\Desktop\\codepy\\googlemodel\\datasets"
model = hub.load(model_path)
movenet = model.signatures['serving_default']
# Load the image to be processed

w1=500
h1=500
# keypoint_names = ['nose', 'left eye', 'right eye', 'left ear', 'right ear', 'left shoulder', 'right shoulder',
#                   'left elbow', 'right elbow', 'left wrist', 'right wrist', 'left hip', 'right hip',
#                   'left knee', 'right knee', 'left ankle', 'right ankle']
# skeleton_connections = [(0, 1), (0, 2), (1, 3), (2, 4), (5, 7), (7, 9), (6, 8), (8, 10), (5, 6), (5, 11),
#                         (6, 12), (11, 13), (13, 15), (12, 14), (14, 16)]

# Define function to draw skeleton connections
def draw_skeleton(image, keypoints, connections):
    for connection in connections:
        start_keypoint = keypoints[connection[0]]
        end_keypoint = keypoints[connection[1]]
        cv2.line(image, (start_keypoint[1], start_keypoint[0]), (end_keypoint[1], end_keypoint[0]), (0, 255, 0), 2)




mouse_state = {'middle': False}
def on_click(x, y, button, pressed):
    if button == mouse.Button.middle:
       mouse_state['middle'] = pressed
with mouse.Listener(on_click=on_click) as listener:  # 创建一个鼠标监听器 
   
 with mss.mss() as sct:
        while True:
         startime=time.time()
         mouse_x, mouse_y = pyautogui.position()
         #print(type(mouse_x))
         top1=int(mouse_y-h1/2)
         left1=int(mouse_x-w1/2)
         monitor = {"top": top1, "left": left1, "width": w1, "height": h1}
                   # 获取屏幕上以鼠标为中心的400*400像素的区域
                  #monitor = {"top": y-h/2, "left": x-w/2, "width": w, "height": h}
         sct_img = sct.grab(monitor)
         img1 = Image.frombytes("RGB", sct_img.size, sct_img.bgra, "raw", "BGRX")
      # 将 PIL 图像转换为 NumPy 数组
         img_np = np.array(img1)
     # 将 BGR 格式转换为 RGB 格式
         img_rgb = cv2.cvtColor(img_np, cv2.COLOR_BGR2RGB)
     # 将 NumPy 数组转换为 TensorFlow 张量
         img_tf = tf.convert_to_tensor(img_rgb)


    #  image0 = tf.io.read_file("test1.jpg")
    #  image1 = tf.compat.v1.image.decode_jpeg(image0)
         original_shape = tf.shape(img_tf)[:2]
         image = tf.expand_dims(img_tf, axis=0)
         image = tf.cast(tf.image.resize_with_pad(image, 192, 192), dtype=tf.int32)

         outputs = movenet(image)
         keypoints = outputs['output_0']



         keypoints = keypoints.numpy().squeeze()
     # 将 keypoints 转换为 NumPy 数组，并提取置信度列
         confidences = keypoints[0:4, 2]
         mean_confidence = np.mean(confidences)
         if mean_confidence<0.3:
            endtime=time.time()
            fps = int(1/(endtime-startime))
            #cv2.putText(img_rgb, 'fps: '+ "%.2f"%(fps), (25, 40), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 255, 0), 2)
            cv2.putText(img_rgb, 'fps: '+ str(fps), (25, 40), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 255, 0), 2)
            cv2.imshow("Image with keypoints", img_rgb)
            cv2.waitKey(1)
            continue
         keypoints = keypoints * np.array([original_shape[0], original_shape[1], 1])
         keypoints = keypoints.astype(np.int32)
         y0, x0 = keypoints[0][:2]
         xre=mouse_x-w1/2+x0
         yre=mouse_y-h1/2+y0

         if mouse_state['middle']:
            pyautogui.click(x=xre, y=yre)   
            
# Convert the TensorFlow image to OpenCV format
    #  image = image1.numpy().squeeze()
    #  image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
#print(keypoints)
         for i in range(keypoints.shape[0]):
             y, x = keypoints[i][:2]
             cv2.circle(img_rgb, (x, y), 3, (0, 0, 255), -1)


           # Draw keypoints and skeleton connections
        #  for  keypoint in enumerate(keypoints):
        #     #  cv2.circle(frame, (int(keypoint[1]), int(keypoint[0])), 3, (0, 0, 255), -1)
        #     #  cv2.putText(frame, keypoint_names[idx], (int(keypoint[1]), int(keypoint[0])), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 1, cv2.LINE_AA)
        #      draw_skeleton(img_rgb, keypoints, skeleton_connections)


         endtime=time.time()
         fps = int(1/(endtime-startime))
         cv2.putText(img_rgb, 'fps: '+ str(fps), (25, 40), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 255, 0), 2)

# Draw the keypoints on the original image
# Display the image
         cv2.imshow("Image with keypoints", img_rgb)
         if cv2.waitKey(1) ==ord('h'):
       
     #    pyautogui.click(x=xre, y=yre, duration=0.1)
 
 
            cv2.destroyAllWindows()
    