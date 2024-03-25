import cv2
import torch
import torchvision.transforms as transforms
from ultralytics import YOLO
import cv2
import datetime
import time
import dlib

import numpy as np


# https://github.com/opencv/opencv/blob/4.x/data/haarcascades/haarcascade_frontalface_default.xml
def haar_face_detect(input_frame):
    """
    Detect faces in an image using Haar Cascade.

    Parameters:
    input_frame (cv2.Mat): Input image in which faces are to be detected.

    Returns:
    cv2.Mat: Image with detected faces marked with rectangles.
    """
    # 加载 Haar Cascade 用于面部检测
    face_cascade_path = "models/haarcascade_frontalface_default.xml"
    face_cascade = cv2.CascadeClassifier(face_cascade_path)

    # 将图像转换为灰度，这是 Haar Cascade 所需的
    gray = cv2.cvtColor(input_frame, cv2.COLOR_BGR2GRAY)

    # 在图像中检测面部
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))
    # 创建用于存储边界框的列表
    bounding_boxes = []

    # 在每个检测到的面部周围绘制矩形框
    for (x, y, w, h) in faces:
        cv2.rectangle(input_frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
        bounding_boxes.append((x, y, w, h))
        face_roi = input_frame[y:y + h, x:x + w]
        # 检查ROI是否为空
        if face_roi.size != 0:
            timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S_%f")
            face_filename = f"face/haar/face_{timestamp}.jpg"
            cv2.imwrite(face_filename, face_roi)
            # 记录照片编号和日期对照表
            with open("face/人臉.txt", "a", encoding="utf-8") as fp_record:
                dt = time.ctime()
                fp_record.write(f"haar檔案: {face_filename} \n")

    return input_frame, bounding_boxes


# Example usage

# input_frame = cv2.imread('test.jpg')
# result_frame = haar_face_detect(input_frame)
# cv2.imshow("Detected Faces", result_frame)
# cv2.waitKey(0)
# cv2.destroyAllWindows()


# https://docs.ultralytics.com/modes/predict/#key-features-of-predict-mode
def pytorch_face_detect(input_frame):
    model_path = r'../models/yolov8_120e.pt'

    # 加载 YOLO 模型
    model = YOLO(model_path)

    # 使用模型进行预测
    results = model([input_frame])
    # boxes2 = []

    # for result in results: #改成在face_recognition算
    #     # 获取边界框和置信度
    #     boxes = result.boxes.xyxy  # 边界框坐标
    #     confs = result.boxes.conf  # 置信度

        # print(f'result:{result}')

        # # 遍历检测到的边界框
        # for box, conf in zip(boxes, confs):
        #     if conf > 0.5:
        #         x1, y1, x2, y2 = map(int, box[:4])
        #         cv2.rectangle(input_frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
        #         detected_faces += 1
        #         boxes2.append((y1, x2, y2, x1))
                # face_roi = input_frame[y1:y2, x1: x2]
                # # 检查ROI是否为空
                # if face_roi.size != 0:
                #     timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S_%f")
                #     face_filename = f"face/yolov8/face_{timestamp}.jpg"
                #     cv2.imwrite(face_filename, face_roi)
                #
                #     # 记录照片编号和日期对照表
                #     with open("face/人臉.txt", "a", encoding="utf-8") as fp_record:
                #         dt = time.ctime()
                #         fp_record.write(f"yolov8檔案: {face_filename} \n")

    return results


# 使用示例

# input_frame = cv2.imread('test.jpg')
# output_frame, faces_detected = pytorch_face_detect(input_frame)
# cv2.imshow('Detected Faces', output_frame)
# cv2.waitKey(0)
# cv2.destroyAllWindows()


def dnn_face_detect(input_frame):
    model_path = "models/opencv_face_detector_uint8.pb"
    config_path = "models/opencv_face_detector.pbtxt"
    # filename = "output"
    net = cv2.dnn.readNetFromTensorflow(model_path, config_path)
    blob = cv2.dnn.blobFromImage(input_frame, 1.0, (300, 300), (104.0, 177.0, 123.0), False, False)
    net.setInput(blob)
    probs = net.forward()
    detected_faces = 0
    box = []

    for i in range(probs.shape[2]):
        confidence = probs[0, 0, i, 2]
        if confidence > 0.5:
            x1 = int(probs[0, 0, i, 3] * input_frame.shape[1])
            y1 = int(probs[0, 0, i, 4] * input_frame.shape[0])
            x2 = int(probs[0, 0, i, 5] * input_frame.shape[1])
            y2 = int(probs[0, 0, i, 6] * input_frame.shape[0])
            box = (x1, y1, x2 - x1, y2 - y1)
            cv2.rectangle(input_frame, (x1, y1), (x2, y2), (0, 0, 255), 2)
            face_roi = input_frame[y1:y2, x1:x2]
            # 检查ROI是否为空
            if face_roi.size != 0:
                timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S_%f")
                face_filename = f"face/dnn/face_{timestamp}.jpg"
                cv2.imwrite(face_filename, face_roi)
                detected_faces += 1

                # 记录照片编号和日期对照表
                with open("face/人臉.txt", "a", encoding="utf-8") as fp_record:
                    dt = time.ctime()
                    fp_record.write(f"dnn檔案: {face_filename} \n")

    return input_frame, detected_faces, box


# # Usage example:
# input_image = cv2.imread("test.jpg")
#
# output_image, face_count, box = dnn_face_detect(input_image)
# print(box)
# cv2.imshow("Detected Faces", output_image)
# cv2.waitKey(0)
# cv2.destroyAllWindows()

def hog_face_detection(image):
    detector = dlib.get_frontal_face_detector()
    # Convert the image to grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Detect faces using the HOG detector
    faces = detector(gray)

    face_rois = []
    boxes = []

    # Process each detected face
    for face in faces:
        x, y, w, h = face.left(), face.top(), face.width(), face.height()
        cv2.rectangle(image, (x, y), (x + w, y + h), (0, 255, 0), 2)

        boxes.append((x, y, w, h))

        # Extract and save the face ROI
        face_roi = image[y:y + h, x:x + w]
        if face_roi.size != 0:
            face_rois.append(face_roi)

            timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S_%f")
            face_filename = f"face/hog/face_{timestamp}.jpg"
            cv2.imwrite(face_filename, face_roi)

            # Log the filename and date
            with open("face/人臉.txt", "a", encoding="utf-8") as fp_record:
                dt = time.ctime()
                fp_record.write(f"hog檔案: {face_filename} \n")

    return image, boxes
    # # Display the result
    # cv2.imshow("HOG Face Detection", image)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()


def cnn_face_detection(image):
    # Create directories for saving faces and logs if they don't exist

    cnn_detector = dlib.cnn_face_detection_model_v1("models/mmod_human_face_detector.dat")

    # Detect faces using the CNN detector
    faces = cnn_detector(image, 1)

    face_rois = []
    boxes = []
    # Process each detected face
    for face in faces:
        x, y, w, h = face.rect.left(), face.rect.top(), face.rect.width(), face.rect.height()
        cv2.rectangle(image, (x, y), (x + w, y + h), (0, 255, 0), 2)

        # Extract and save the face ROI
        face_roi = image[y:y + h, x:x + w]
        if face_roi.size != 0:
            # 截取面部区域
            face_rois.append(face_roi)
            boxes.append((x, y, w, h))

            timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S_%f")
            face_filename = f"face/cnn/face_{timestamp}.jpg"
            cv2.imwrite(face_filename, face_roi)

            # Log the filename and date
            with open("face/人臉.txt", "a", encoding="utf-8") as fp_record:
                dt = time.ctime()
                fp_record.write(f"cnn檔案: {face_filename} \n")

    return image, boxes
    # # Display the result
    # cv2.imshow("CNN Face Detection", image)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()

# image_path = "test.jpg"
# image = cv2.imread(image_path)
# image, box = hog_face_detection(image)
# cv2.imshow("CNN Face Detection", image)
