import cv2
import numpy as np
import time
from model import dnn_face_detect, pytorch_face_detect, haar_face_detect, hog_face_detection, cnn_face_detection
from face__recognition import load_img,load_csv, face_rec, deep_face
import datetime
import pandas as pd
import os

known_face_encodings = []
import math

# from playsound import playsound
frame_count = 0
key = -1
capture = None  # Start capturing
video_frame = None
i, amount = 1, 1
hickness = 2  # Rectangle line size
thickness = 2  # Rectangle line size
shift = 0  # Rectangle size, 0:normal
prev_gray, gray, flow, cflow, frame = None, None, None, None, None
# motion2color, motion2color_copy_res, motion2color_copy_dgr, motion2color_copy_vis = None, None, None, None
draw_image = None
camera = None
input_image2_mat = None
src, thr, dst = None, None, None
squares = []
contours = []  # Vector for storing contour
hierarchy = []
approx = []
max_cosine = 0
cosine = 0
p = None
fp_dgr, fp_vis, fp_res, fp_record = None, None, None, None
video_frame_mat, input_image_roi_mat_copy = None, None

duration = 0.0
tstart, tfinish, save_tstart, save_tfinish = None, None, None, None


class Res:
    def __init__(self):
        self.iwidth_res = 0
        self.iheight_res = 0
        self.input_image_roi_res = None
        self.show_roi_res = False
        self.color_res = (255, 0, 0)
        self.vertex_one_res = (0, 0)
        self.vertex_three_res = (0, 0)
        self.save_name = ""
        self.csave_filename_res = ""
        self.image_count_res = 1
        self.input_image2_mat = None
        self.input_img = None
        # 其他可能需要的类属性

    def roi_part_res(self):
        global frame, draw_image

        cv2.rectangle(draw_image, tuple(self.vertex_one_res), tuple(self.vertex_three_res), self.color_res, thickness,
                      cv2.LINE_AA, shift)
        # cv2.imshow("drawImage", draw_image)

        self.iwidth_res = self.vertex_three_res[0] - self.vertex_one_res[0]
        self.iheight_res = self.vertex_three_res[1] - self.vertex_one_res[1]

        self.input_img = np.zeros_like(frame, dtype=np.uint8)

        if self.input_image_roi_res is not None:
            self.input_img = self.input_image_roi_res.copy()

        if self.iwidth_res != 0 and self.iheight_res != 0:
            x1, y1, w, h = self.vertex_one_res[0], self.vertex_one_res[1], abs(self.iwidth_res), abs(self.iheight_res)

            # 根据 iwidth_res 和 iheight_res 的正负来调整 ROI 的起始点
            if self.iwidth_res < 0:
                x1 = x1 - w
            if self.iheight_res < 0:
                y1 = y1 - h

            # 计算 ROI 的结束点
            x2, y2 = x1 + w, y1 + h

            if self.input_image_roi_res is not None:
                self.input_image_roi_res = self.input_image_roi_res[y1:y2, x1:x2].copy()

            if self.input_img is not None:
                self.input_img = self.input_img[y1:y2, x1:x2].copy()

            self.input_image2_mat = self.input_img
            self.input_image2_mat = self.input_img  # 共享相同的数据
            findsquares(self.input_image2_mat)

     ㄙㄩ   if self.show_roi_res and self.input_image_roi_res is not None:
            cv2.imshow("inputImageROI_RES", self.input_img)

    def on_mouse_res(self, event_res, x_res, y_res, flags_res, param_res):
        global amount
        if event_res == cv2.EVENT_LBUTTONDOWN or event_res == cv2.EVENT_RBUTTONDOWN:
            self.vertex_one_res = (x_res, y_res)

        if event_res == cv2.EVENT_LBUTTONUP or event_res == cv2.EVENT_RBUTTONUP:
            self.vertex_three_res = (x_res, y_res)
            self.show_roi_res = True
            # 注意：这里的 'amount' 变量需要被定义在类中或者作为参数传入
            amount += 1

        if flags_res == cv2.EVENT_FLAG_LBUTTON or flags_res == cv2.EVENT_FLAG_RBUTTON:
            self.vertex_three_res = (x_res, y_res)


class Dgr:
    def __init__(self):
        self.iwidth_dgr = 0
        self.iheight_dgr = 0
        self.high_roi_dgr = None
        self.show_roi_dgr = False
        self.color_dgr = (0, 0, 255)
        self.vertex_one_dgr = (0, 0)
        self.vertex_three_dgr = (0, 0)
        self.csave_filename_dgr = ""
        self.image_count_dgr = 1
        self.count_aver_dgr = 0.0
        self.aver_dgr = 0.0

    def roi_part_dgr(self):
        global draw_image, frame

        cv2.rectangle(draw_image, self.vertex_one_dgr, self.vertex_three_dgr, self.color_dgr, thickness, cv2.LINE_AA,
                      shift)
        cv2.imshow("draw_image", draw_image)

        self.iwidth_dgr = self.vertex_three_dgr[0] - self.vertex_one_dgr[0]
        self.iheight_dgr = self.vertex_three_dgr[1] - self.vertex_one_dgr[1]

        self.input_img = np.zeros_like(frame, dtype=np.uint8)

        if self.input_image_roi_dgr is not None:
            self.input_img = self.input_image_roi_dgr.copy()

        if self.iwidth_dgr != 0 and self.iheight_dgr != 0:
            x1, y1 = self.vertex_one_dgr
            w, h = abs(self.iwidth_dgr), abs(self.iheight_dgr)

            # 根据 iwidth_dgr 和 iheight_dgr 的正负来调整 ROI 的起始点
            if self.iwidth_dgr < 0:
                x1 = x1 - w
            if self.iheight_dgr < 0:
                y1 = y1 - h

            # 计算 ROI 的结束点
            x2, y2 = x1 + w, y1 + h

            if self.high_roi_dgr is not None:
                # 切片以提取 ROI
                self.high_roi_dgr = self.high_roi_dgr[y1:y2, x1:x2].copy()

        if self.show_roi_dgr and self.high_roi_dgr is not None:
            cv2.imshow("input_image_roi_dgr", self.high_roi_dgr)

    def on_mouse_dgr(self, event_dgr, x_dgr, y_dgr, flags_dgr, param_dgr):
        global amount
        if event_dgr == cv2.EVENT_LBUTTONDOWN or event_dgr == cv2.EVENT_RBUTTONDOWN:
            self.vertex_one_dgr = (x_dgr, y_dgr)

        if event_dgr == cv2.EVENT_LBUTTONUP or event_dgr == cv2.EVENT_RBUTTONUP:
            self.vertex_three_dgr = (x_dgr, y_dgr)
            self.show_roi_dgr = True
            amount += 1

        if flags_dgr == cv2.EVENT_FLAG_LBUTTON or flags_dgr == cv2.EVENT_FLAG_RBUTTON:
            self.vertex_three_dgr = (x_dgr, y_dgr)


class Vis:
    def __init__(self):
        self.iwidth_vis = 0
        self.iheight_vis = 0
        self.show_roi_vis = False
        self.color_vis = (0, 255, 0)
        self.vertex_one_vis = (0, 0)
        self.vertex_three_vis = (0, 0)
        self.csave_filename_vis = ""
        self.image_count_vis = 1
        self.count_aver_vis = 0.0
        self.aver_vis = 0.0
        self.high_roi_vis = None
        self.input_image_roi_vis = None
        self.known_face_encodings = load_img()
        self.result_csv = load_csv()
        self.last_save_time = datetime.datetime.now()

    def roi_part_vis(self):
        global draw_image
        # 检查 vertex_one_vis 和 vertex_three_vis 是否已经被设置
        if self.vertex_one_vis is not None and self.vertex_three_vis is not None:
            # 绘制矩形
            cv2.rectangle(draw_image, self.vertex_one_vis, self.vertex_three_vis, self.color_vis, thickness,
                          cv2.LINE_AA, shift)
            # cv2.imshow("test", draw_image)
            self.input_image_roi_vis = draw_image.copy()

            self.iwidth_vis = self.vertex_three_vis[0] - self.vertex_one_vis[0]
            self.iheight_vis = self.vertex_three_vis[1] - self.vertex_one_vis[1]

            if self.iwidth_vis != 0 and self.iheight_vis != 0:
                x1, y1 = self.vertex_one_vis
                w, h = abs(self.iwidth_vis), abs(self.iheight_vis)

                # 根据 iwidth_vis 和 iheight_vis 的正负来调整 ROI 的起始点
                if self.iwidth_vis < 0:
                    x1 = x1 - w
                if self.iheight_vis < 0:
                    y1 = y1 - h

                # 计算 ROI 的结束点
                x2, y2 = x1 + w, y1 + h

                if self.high_roi_vis is not None:
                    # 切片以提取 ROI
                    self.high_roi_vis = self.high_roi_vis[y1:y2, x1:x2].copy()
                    self.input_image_roi_vis = self.input_image_roi_vis[y1:y2, x1:x2].copy()

        if self.show_roi_vis and self.high_roi_vis is not None:

            # dnn_detect, dnn_detected_faces, box = dnn_face_detect(self.input_image_roi_vis)
            # cv2.imshow("dnn_detect", dnn_detect)
            # cv2.moveWindow("dnn_detect", 1000, 50)

            if frame_count % 10 == 0:
                self.result_csv = face_rec(self.input_image_roi_vis, self.known_face_encodings, self.result_csv, self.last_save_time)
                pass
                # yolov8_detect, box2 = pytorch_face_detect(self.input_image_roi_vis)
                # cv2.imshow("yolov8_detect", yolov8_detect)
                # cv2.moveWindow("yolov8_detect", 1000, 400)
            #
            # haar_detect, box3 = haar_face_detect(self.input_image_roi_vis)
            # cv2.imshow("haar_detect", haar_detect)
            # cv2.moveWindow("haar_detect", 1000, 600)

            # cnn_detect, box4 = cnn_face_detection(self.input_image_roi_vis)
            # cv2.imshow("cnn_detect", cnn_detect)
            # cv2.moveWindow("cnn_detect", 500, 400)

            # hog_detect, box5 = hog_face_detection(self.input_image_roi_vis)
            # cv2.imshow("hog_detect", hog_detect)
            # cv2.moveWindow("hog_detect", 500, 600)

    def on_mouse_vis(self, event_vis, x_vis, y_vis, flags_vis, param_vis):
        global amount, tstart, save_tstart
        if event_vis == cv2.EVENT_LBUTTONDOWN or event_vis == cv2.EVENT_RBUTTONDOWN:
            self.vertex_one_vis = (x_vis, y_vis)

        if event_vis == cv2.EVENT_LBUTTONUP or event_vis == cv2.EVENT_RBUTTONUP:
            self.vertex_three_vis = (x_vis, y_vis)
            self.show_roi_vis = True

            amount += 1

        if flags_vis == cv2.EVENT_FLAG_LBUTTON or flags_vis == cv2.EVENT_FLAG_RBUTTON:
            self.vertex_three_vis = (x_vis, y_vis)
            tstart = time.time()
            save_tstart = time.time()


def turn_and_count(vis_instance, dgr_instance):
    global duration, tfinish, tstart, frame

    printB = 0

    if duration < 60:
        tfinish = time.time()
        duration = tfinish - tstart
        # print(f'Duration: {duration}')

    # 处理高危区域 (DGR)
    high_roi_dgr_mat = None
    if len(dgr_instance.high_roi_dgr.shape) == 3 and dgr_instance.high_roi_dgr.shape[2] == 3:
        high_roi_dgr_mat = cv2.cvtColor(dgr_instance.high_roi_dgr, cv2.COLOR_BGR2GRAY).copy()
    else:
        high_roi_dgr_mat = dgr_instance.high_roi_dgr.copy()

    n_white_count_dgr, n_black_count_dgr = pixel_counter(high_roi_dgr_mat)
    # print(f"high_roi_dgr_mat: {type(high_roi_dgr_mat)} , {high_roi_dgr_mat.shape}")
    # print(f"n_black_count_dgr: {type(n_black_count_dgr)}")
    percent_dgr = n_white_count_dgr / (n_black_count_dgr + n_white_count_dgr)

    if printB:
        print("\n危險事件roi(紅色矩形)像素個數統計:")
        print(
            f"白像素: {n_white_count_dgr}\n黑像素: {n_black_count_dgr}\n總像素: {n_black_count_dgr + n_white_count_dgr}")
        print(f"危險事件偵測動量: {percent_dgr}")

    # DGR的保存逻辑
    if percent_dgr > 0.0 and duration >= 60.0:
        if dgr_instance.count_aver_dgr < 10:
            dgr_instance.aver_dgr += percent_dgr
            dgr_instance.count_aver_dgr += 1
            if dgr_instance.count_aver_dgr <= 9:
                dgr_instance.aver_dgr = dgr_instance.aver_dgr / (dgr_instance.count_aver_dgr + 1)
        else:
            if dgr_instance.aver_dgr - (
                    0.1 * dgr_instance.aver_dgr) <= percent_dgr <= dgr_instance.aver_dgr + (
                    0.1 * dgr_instance.aver_dgr):
                dgr_instance.csave_filename_dgr = f"dangerous record/normal_{dgr_instance.image_count_dgr}.jpg"
                cv2.imwrite(dgr_instance.csave_filename_dgr, frame)
                dgr_instance.csave_filename_dgr = f"dangerous record/active_{dgr_instance.image_count_dgr}.jpg"
                cv2.imwrite(dgr_instance.csave_filename_dgr, high_roi_dgr_mat)

                dgr_instance.image_count_dgr += 1

    # print(f"危險事件偵測平均動量: {dgr_instance.aver_dgr}")

    # 处理社交访客区域 (VIS)
    high_roi_vis_mat = None
    if vis_instance.high_roi_vis is not None and len(vis_instance.high_roi_vis.shape) == 3:
        high_roi_vis_mat = cv2.cvtColor(vis_instance.high_roi_vis, cv2.COLOR_BGR2GRAY).copy()
    else:
        # 如果 high_roi_vis 不是彩色图像，直接使用它或者处理错误情况
        high_roi_vis_mat = vis_instance.high_roi_vis.copy() if vis_instance.high_roi_vis is not None else None

    n_white_count_vis, n_black_count_vis = pixel_counter(high_roi_vis_mat)
    percent_vis = n_white_count_vis / (n_black_count_vis + n_white_count_vis)

    if printB:
        print("\n社交訪客roi(藍色矩形)像素個數統計:")
        print(
            f"白像素: {n_white_count_vis}\n黑像素: {n_black_count_vis}\n總像素: {n_black_count_vis + n_white_count_vis}")
        print(f"社交訪客偵測動量: {percent_vis}")

    if percent_vis > 0.0 and duration >= 10.0:

        # test
        # print("\n社交訪客roi(藍色矩形)像素個數統計:")
        # print( f"白像素: {n_white_count_vis}\n黑像素: {n_black_count_vis}\n總像素: {n_black_count_vis + n_white_count_vis}")
        # print(f"社交訪客偵測動量: {percent_vis}")
        # print(f"vis_instance.aver_vis: {vis_instance.aver_vis}")

        # try:
        #     vis_instance.csave_filename_vis = f"visitor/active_{vis_instance.image_count_vis}.jpg"
        #     is_saved = cv2.imwrite(vis_instance.csave_filename_vis, high_roi_vis_mat)
        #     if not is_saved:
        #         raise ValueError("圖像無法保存。")
        #     print("圖像保存成功。")
        # except Exception as e:
        #     print(f"圖像保存出現錯誤：{e}")

        if vis_instance.count_aver_vis < 10:
            # print(f"vis_instance.aver_vis<10: {vis_instance.aver_vis}")
            vis_instance.aver_vis += percent_vis
            vis_instance.count_aver_vis += 1
            if vis_instance.count_aver_vis <= 9:
                vis_instance.aver_vis = vis_instance.aver_vis / (vis_instance.count_aver_vis + 1)


        else:
            if vis_instance.aver_vis - (
                    0.1 * vis_instance.aver_vis) <= percent_vis <= vis_instance.aver_vis + (
                    0.1 * vis_instance.aver_vis):
                # print(f"vis_instance.aver_vis>10: {vis_instance.aver_vis}")
                # print(f"percent_vis: {percent_vis}")
                vis_instance.csave_filename_vis = f"visit history/normal_{vis_instance.image_count_vis}.jpg"
                cv2.imwrite(vis_instance.csave_filename_vis, frame)
                vis_instance.csave_filename_vis = f"visit history/active_{vis_instance.image_count_vis}.jpg"
                cv2.imwrite(vis_instance.csave_filename_vis, high_roi_vis_mat)

                vis_instance.image_count_vis += 1

    # print(f"社交訪客偵測平均動量: {vis_instance.aver_vis}")


def pixel_counter(src):
    # Ensure the source image is single-channel
    if len(src.shape) != 2:
        raise ValueError("PixelCounter: The source image must be single-channel.")

    # Count of white and black pixels
    count_white = np.sum(src > 0)
    count_black = np.sum(src == 0)

    # Return count based on nflag

    return count_white, count_black


# approxPolyDP 函数返回的 approx 数组是一个包含点坐标的数组，其中每个点坐标是一个形状为 (1, 2) 的 NumPy 数组
def angle(pt1, pt2, pt0):
    dx1 = pt1[0][0] - pt0[0][0]
    dy1 = pt1[0][1] - pt0[0][1]
    dx2 = pt2[0][0] - pt0[0][0]
    dy2 = pt2[0][1] - pt0[0][1]
    return (dx1 * dx2 + dy1 * dy2) / np.sqrt((dx1 * dx1 + dy1 * dy1) * (dx2 * dx2 + dy2 * dy2) + 1e-10)


def findsquares(image):
    global squares, max_cosine, cosine, p, image_count_rec, csave_filename_rec, save_tstart, save_tfinish, thr, dst, src, gray, contours, hierarchy, approx

    src = image.copy()  # Load source image
    thr = np.zeros((src.shape[0], src.shape[1]), dtype=np.uint8)
    dst = np.zeros_like(src)

    gray = cv2.cvtColor(src, cv2.COLOR_BGR2GRAY)  # Convert to gray
    _, thr = cv2.threshold(gray, 150, 255, cv2.THRESH_BINARY)  # Threshold the gray

    squares = []

    contours, hierarchy = cv2.findContours(thr, cv2.RETR_CCOMP,
                                           cv2.CHAIN_APPROX_SIMPLE)  # Find the contours in the image
    color = (0, 0, 255)

    for i in range(len(contours)):  # Iterate through each contour
        epsilon = 0.02 * cv2.arcLength(contours[i], True)
        approx = cv2.approxPolyDP(contours[i], epsilon, True)

        # Calculate the 4 vertices of the rectangle after calculating the contour area
        if len(approx) == 4 and abs(cv2.contourArea(approx)) > 1000 and cv2.isContourConvex(approx):
            max_cosine = 0
            for j in range(2, 5):
                # Calculate the maximum cosine of the angles between contour edges
                cosine = abs(angle(approx[j % 4], approx[j - 2], approx[j - 1]))
                max_cosine = max(max_cosine, cosine)

            if max_cosine < 0.3:
                squares.append(approx)

    for i in range(len(squares)):
        p = squares[i].reshape(-1, 2)

        n = len(squares[i])
        if p[0, 0] > 3 and p[0, 1] > 3:
            cv2.polylines(src, [p], isClosed=True, color=(0, 255, 0), thickness=3)
            save_tfinish = time.time()
            if (save_tfinish - save_tstart) >= 60.0:
                csave_filename_rec = f"social interaction record/{image_count_rec}.jpg"
                cv2.imwrite(csave_filename_rec, src)
                csave_filename_rec = f"social interaction record/{image_count_rec}_ROI.jpg"
                cv2.imwrite(csave_filename_rec, src)
                image_count_rec += 1

                with open("social interaction record/社交互動.txt", "a") as fp_rec:
                    now = time.time()
                    fp_rec.write(f"偵測到社交互動事件 {now}")

                save_tstart = time.time()  # Reset the timer

    # cv2.imshow("dst", src)
    # cv2.imshow("Thread", thr)
    # cv2.imshow("Source", src)
    # cv2.imshow("Contour", dst)


def main():
    global amount, frame, key, gray, draw_image, frame_count

    # 创建类的实例
    dgr_instance = Dgr()
    res_instance = Res()
    vis_instance = Vis()

    # 初始化摄像头
    cap = cv2.VideoCapture(0)
    if cap.isOpened():
        ret, frame = cap.read()
        if not ret or frame is None:
            print("Could not initialize...")
            exit(-1)
    else:
        print("Could not initialize...")
        exit(-1)

    # 创建背景减除器
    bg_subtractor = cv2.createBackgroundSubtractorMOG2()
    bg_subtractor.setVarThreshold(16)
    bg_subtractor.setDetectShadows(True)

    fg_mask = None

    while True:
        ret, frame = cap.read()
        if not ret or frame is None:
            break

        frame_count += 1

        fg_mask = bg_subtractor.apply(frame, fg_mask)

        # # 复制帧到高亮区域（DGR 和 VIS）
        dgr_instance.high_roi_dgr = fg_mask.copy()
        vis_instance.high_roi_vis = fg_mask.copy()

        # 复制帧到输入图像 ROI（REC、DGR 和 VIS）
        res_instance.input_image_roi_res = frame.copy()
        dgr_instance.input_image_roi_dgr = frame.copy()
        vis_instance.input_image_roi_vis = frame.copy()

        # 转换帧到灰度图像
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        cv2.namedWindow("draw_image")
        cv2.moveWindow("draw_image", 50, 100)

        if amount == 1:

            draw_image = frame.copy()
            cv2.setMouseCallback("draw_image", res_instance.on_mouse_res, param=res_instance)
            res_instance.roi_part_res()  # 传递 res_instance 到相关函数

        elif amount == 2:
            res_instance.roi_part_res()
            draw_image = frame.copy()
            cv2.setMouseCallback("draw_image", dgr_instance.on_mouse_dgr, param=dgr_instance)
            dgr_instance.roi_part_dgr()  # 传递 dgr_instance 到相关函数

        elif amount == 3:

            res_instance.roi_part_res()
            dgr_instance.roi_part_dgr()
            draw_image = frame.copy()
            cv2.setMouseCallback("draw_image", vis_instance.on_mouse_vis)
            vis_instance.roi_part_vis()


        else:

            draw_image = frame.copy()

            res_instance.roi_part_res()
            dgr_instance.roi_part_dgr()
            vis_instance.roi_part_vis()

            if amount == 4:
                # 获取当前时间戳
                st = int(time.time())
                save_name = f"框选结果/{st}.jpg"
                # 保存绘图图像
                cv2.imwrite(save_name, draw_image)
                # 记录照片编号和日期对照表
                with open("框选结果/照片编号日期对照表.txt", "a", encoding="utf-8") as fp_record:
                    dt = time.ctime()
                    fp_record.write(f"编号 {st} 时间点: {dt}\n")
                amount += 1
            # overflow()
            # turn_and_count(vis_instance, dgr_instance)  # 传递 vis_instance 和 dgr_instance 到相关函数 (人臉辨識先暫停)

        # 显示图像窗口
        cv2.imshow("draw_image", draw_image)

        key = cv2.waitKey(10)
        if key == 27:
            break

    # 释放资源
    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
