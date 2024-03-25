import cv2
import face_recognition
import numpy as np
from deepface import DeepFace
from PIL import ImageFont, ImageDraw, Image
from model import pytorch_face_detect
import datetime
import pandas as pd
import os
import glob

def encode_faces(base_path="datasets/NewJeans/"):
    known_face_paths = {}
    # 遍歷 base_path 下的每個子目錄
    for person_name in os.listdir(base_path):
        person_path = os.path.join(base_path, person_name)
        # 確保是目錄
        if os.path.isdir(person_path):
            # 使用 glob 模塊搜集每個子目錄下的 jpg 文件
            jpg_files = glob.glob(os.path.join(person_path, "*.jpg"))
            # 如果有 jpg 文件，則添加到 known_face_paths 字典中
            if jpg_files:
                known_face_paths[person_name] = jpg_files
                print(f'正在讀入:{jpg_files}')
    return known_face_paths

def load_img():
    print('計算人臉數據庫關鍵點中...')
    # 更新你的人臉數據庫路徑和名稱，包括每個人的多張圖片
    # known_face_paths = {
    #     "Hyein": [
    #         "datasets/NewJeans/Hyein/Hyein0.jpg",
    #         "datasets/NewJeans/Hyein/Hyein1.jpg",
    #         "datasets/NewJeans/Hyein/Hyein2.jpg",
    #         "datasets/NewJeans/Hyein/Hyein3.jpg",
    #         "datasets/NewJeans/Hyein/Hyein4.jpg"
    #     ],
    #     "Minji": [
    #         "datasets/NewJeans/Minji/Minji0.jpg",
    #         "datasets/NewJeans/Minji/Minji1.jpg",
    #         "datasets/NewJeans/Minji/Minji2.jpg",
    #         "datasets/NewJeans/Minji/Minji3.jpg",
    #         "datasets/NewJeans/Minji/Minji4.jpg"
    #     ],
    #     "Hanni": [
    #         "datasets/NewJeans/Hanni/Hanni0.jpg",
    #         "datasets/NewJeans/Hanni/Hanni1.jpg",
    #         "datasets/NewJeans/Hanni/Hanni2.jpg",
    #         "datasets/NewJeans/Hanni/Hanni3.jpg",
    #         "datasets/NewJeans/Hanni/Hanni4.jpg"
    #     ],
    #     "Danielle": [
    #         "datasets/NewJeans/Danielle/NewJeansDanielle0.jpg",
    #         "datasets/NewJeans/Danielle/NewJeansDanielle1.jpg",
    #         "datasets/NewJeans/Danielle/NewJeansDanielle2.jpg",
    #         "datasets/NewJeans/Danielle/NewJeansDanielle3.jpg",
    #         "datasets/NewJeans/Danielle/NewJeansDanielle4.jpg"
    #     ],
    #     "Haerin": [
    #         "datasets/NewJeans/Haerin/Haerin0.jpg",
    #         "datasets/NewJeans/Haerin/Haerin1.jpg",
    #         "datasets/NewJeans/Haerin/Haerin2.jpg",
    #         "datasets/NewJeans/Haerin/Haerin3.jpg",
    #         "datasets/NewJeans/Haerin/Haerin4.jpg"
    #     ]
    # }
    known_face_paths = encode_faces()
    known_face_encodings = []

    # 加載每個人的所有臉部圖像
    for name, face_paths in known_face_paths.items():
        for face_path in face_paths:
            try:
                image = face_recognition.load_image_file(face_path)
                encoding = face_recognition.face_encodings(image)[0]
                known_face_encodings.append((name, encoding))
            except IndexError:
                print(f"在 {face_path} 中未檢測到面部")
    print('計算完成')

    return known_face_encodings


def load_csv():
    # # # 检查之前的结果文件是否存在
    results_file = 'result.csv'
    if os.path.exists(results_file):
        # 如果存在，读取之前的结果
        print('載入Dataframe')
        results_df = pd.read_csv(results_file)
    else:
        # 否则，初始化一个空的 DataFrame
        print('初始化Dataframe')
        results_df = pd.DataFrame()

    return results_df


def face_rec(frame, known_face_encodings, result_csv, last_save_time):
    # 將圖像從BGR轉換為RGB
    rgb_frame = np.ascontiguousarray(frame[:, :, ::-1])

    # # 檢測圖像中的所有面部(HOG)
    # face_locations = face_recognition.face_locations(rgb_frame)
    # face_encodings = face_recognition.face_encodings(rgb_frame, face_locations)

    # 使用CNN模型进行面部检测
    yolov8_results = pytorch_face_detect(rgb_frame)
    face_locations = []

    for result in yolov8_results:
        boxes = result.boxes.xyxy  # 边界框坐标
        confs = result.boxes.conf  # 置信度
        # 遍历检测到的边界框
        for box, conf in zip(boxes, confs):
            if conf > 0.6:
                x1, y1, x2, y2 = map(int, box[:4])
                face_locations.append((y1, x2, y2, x1))
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)

    face_encodings = face_recognition.face_encodings(rgb_frame, face_locations)

    for result, (top, right, bottom, left), face_encoding in zip(yolov8_results, face_locations, face_encodings):
        name = "Unknown"

        # matches = face_recognition.compare_faces([encoding for name, encoding in known_face_encodings], face_encoding)
        #
        # if True in matches:
        #     first_match_index = matches.index(True)
        #     name = known_face_encodings[first_match_index][0]
        min_distance = 0.6
        # 比較已知的面部編碼
        for known_name, known_encoding in known_face_encodings:
            distance = np.linalg.norm(known_encoding - face_encoding)
            if distance < min_distance:
                min_distance = distance
                name = known_name

        face_roi = frame[top:bottom, left:right]

        filename = None
        # 检查ROI是否为空
        if face_roi.size != 0:
            timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S_%f")
            face_filename = f"{name}_{timestamp}.jpg"
            filename = face_filename
            file_path = os.path.join("face/yolov8", face_filename)
            cv2.imwrite(file_path, face_roi)

        # 在圖像中標記面部
        cv2.rectangle(frame, (left, top), (right, bottom), (0, 255, 0), 2)
        cv2.putText(frame, name, (left + 6, bottom - 6), cv2.FONT_HERSHEY_DUPLEX, 0.5, (255, 255, 255), 1)

        result_csv = deep_face(rgb_frame, face_locations, filename, result_csv, result, name)

        current_time = datetime.datetime.now()
        # 检查是否超过5分钟，如果是，则保存CSV
        if (current_time - last_save_time).total_seconds() > 5:
            csv_filename = "result.csv"
            result_csv.to_csv(csv_filename, index=False)
            print(f"Results saved to {csv_filename}")

            # 更新上次保存时间
            last_save_time = current_time

    # 顯示結果圖像
    cv2.imshow('Face Recognition', frame)
    cv2.moveWindow("Face Recognition", 800, 400)

    return result_csv


# 定義加入文字函式
def putText(img, x, y, text, size=50, color=(255, 255, 255)):
    fontpath = 'font/NotoSansCJKtc-Regular.otf'  # 字体
    font = ImageFont.truetype(fontpath, size)  # 定义字体和文字大小
    imgPil = Image.fromarray(img)  # 转换成 PIL 图像对象
    draw = ImageDraw.Draw(imgPil)  # 定义绘图对象
    draw.text((x, y), text, fill=color, font=font)  # 加入文字
    img = np.array(imgPil)  # 转换成 np.array
    return img


def deep_face(frame, face_locations, face_filename, df, result, name):
    # face_locations = pytorch_face_detect(frame)

    preprocess_speed = result.speed['preprocess']
    inference_speed = result.speed['inference']
    postprocess_speed = result.speed['postprocess']
    total_speed = preprocess_speed + inference_speed + postprocess_speed

    orig_shape_str = str(result.orig_shape)  # 将形状转换为字符串
    # 调整 face_locations 处理逻辑
    for (top, right, bottom, left) in face_locations:
        # 注意这里的坐标转换，确保提取的是正确的人脸区域

        # 提取单个人脸区域
        face = frame[top:bottom, left:right]
        cv2.imshow('Emotion Analysis', face)
        try:
            # 对单个人脸区域进行 DeepFace 情绪分析
            analyses = DeepFace.analyze(frame[top:bottom, left:right],
                                        enforce_detection=False)  # 确保传递正确的人脸区域
            for analyze in analyses:
                # dominant_emotion = analyze['dominant_emotion']
                # x, y, w, h = analyze['region']['x'], analyze['region']['y'], analyze['region']['w'], analyze['region'][
                #     'h']
                print(analyze)
                # 在图像中标记人脸
                # cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
                # # 在人脸上方显示情绪
                # cv2.putText(frame, dominant_emotion, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

                data = {
                    'face_filename': face_filename,
                    'name': name,
                    "Dominant Emotion": analyze['dominant_emotion'],
                    "Happy": analyze['emotion']['happy'],
                    "Neutral": analyze['emotion']['neutral'],
                    "Angry": analyze['emotion']['angry'],
                    "Disgust": analyze['emotion']['disgust'],
                    "Fear": analyze['emotion']['fear'],
                    "Sad": analyze['emotion']['sad'],
                    "Surprise": analyze['emotion']['surprise'],
                    "Age": analyze['age'],
                    "Dominant Gender": analyze['dominant_gender'],
                    "Dominant Race": analyze['dominant_race'],
                    "Face Confidence": analyze['face_confidence'],
                    # yolov8 inference detail:
                    'orig_shape': orig_shape_str,
                    'preprocess_speed': preprocess_speed,
                    'inference_speed': inference_speed,
                    'postprocess_speed': postprocess_speed,
                    'total_speed': total_speed
                }

                # 将结果添加到 DataFrame 中
                df = pd.concat([df, pd.DataFrame([data])], ignore_index=True)

        except Exception as e:
            print("DeepFace 分析錯誤:", e)
            data = {
                "DeepFace 分析錯誤:", e
            }
            df = pd.concat([df, pd.DataFrame([data])], ignore_index=True)

    cv2.imshow('deep face', frame)
    cv2.moveWindow("deep face", 500, 200)

    return df

# known_face_paths = load_img()
#
# # 打開攝像頭
# cap = cv2.VideoCapture(0)
#
# while True:
#     ret, frame = cap.read()
#     if not ret:
#         print("未能獲取攝像頭圖像")
#         break
#
#     # 將圖像從BGR轉換為RGB
#     rgb_frame = np.ascontiguousarray(frame[:, :, ::-1])
#
#     # 檢測圖像中的所有面部
#     face_locations = face_recognition.face_locations(rgb_frame)
#     face_encodings = face_recognition.face_encodings(rgb_frame, face_locations)
#
#     for (top, right, bottom, left), face_encoding in zip(face_locations, face_encodings):
#         name = "Unknown"
#         # matches = face_recognition.compare_faces([encoding for name, encoding in known_face_encodings], face_encoding)
#         #
#         # if True in matches:
#         #     first_match_index = matches.index(True)
#         #     name = known_face_encodings[first_match_index][0]
#         min_distance = 0.5
#         # 比較已知的面部編碼
#         for known_name, known_encoding in known_face_encodings:
#             distance = np.linalg.norm(known_encoding - face_encoding)
#             if distance < min_distance:
#                 min_distance = distance
#                 name = known_name
#
#         # 在圖像中標記面部
#         cv2.rectangle(frame, (left, top), (right, bottom), (0, 255, 0), 2)
#         cv2.putText(frame, name, (left + 6, bottom - 6), cv2.FONT_HERSHEY_DUPLEX, 0.5, (255, 255, 255), 1)
#
#     # 顯示結果圖像
#     cv2.imshow('Face Recognition', frame)
#
#     # 按 'q' 退出循環
#     if cv2.waitKey(1) & 0xFF == ord('q'):
#         break
#
# # 釋放攝像頭資源
# cap.release()
# cv2.destroyAllWindows()
