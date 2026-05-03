#!/usr/bin/env python3

import cv2
import rospy
import os
import threading
import numpy as np
import onnxruntime as ort
from typing import List, Optional
from dataclasses import dataclass
from sensor_msgs.msg import Image
from PIL import Image as PILImage, ImageDraw, ImageFont 
from robot_navigation.srv import *
from cv_bridge import CvBridge


@dataclass
class DetectionResult:
    x: int
    y: int
    w: int
    h: int
    label: str
    score: float


class MultipleDetection:
    def __init__(self, onnx_model: str, confidence_thres: float, iou_thres: float, class_names: List[str]):
        rospy.init_node('person_recognition_service')
        
        # ROS服务
        self.service = rospy.Service('/recognize_person', detect, self.handle_recognition)  

         # 图像订阅和发布
        self.bridge = CvBridge()
        self.latest_image = None
        self.image_lock = threading.Lock()
        self.image_sub = rospy.Subscriber("/image_raw", Image, self.image_callback)
        self.img_save_path1 = "/home/zcy/simulation_ws/src/robot_navigation/picture/person1.jpg"
        self.img_save_path2 = "/home/zcy/simulation_ws/src/robot_navigation/picture/person2.jpg"

        # 结果图像发布
        self.result_pub = rospy.Publisher("/recognized_image", Image, queue_size=1) 
        self.detected_img_save_path1 = "/home/zcy/simulation_ws/src/robot_navigation/scripts/pic/person1.jpg"
        self.detected_img_save_path2 = "/home/zcy/simulation_ws/src/robot_navigation/scripts/pic/person2.jpg"

        self.onnx_model = onnx_model
        self.confidence_thres = confidence_thres
        self.iou_thres = iou_thres
        self.classes = class_names

        available_providers = ort.get_available_providers()
        if "CUDAExecutionProvider" in available_providers:
            providers = ["CUDAExecutionProvider", "CPUExecutionProvider"]
        else:
            providers = ["CPUExecutionProvider"]

        self.session = ort.InferenceSession(self.onnx_model, providers=providers)
        input_shape = self.session.get_inputs()[0].shape
        self.input_width, self.input_height = input_shape[2], input_shape[3]

        rospy.loginfo("person recognition service is ready")

    # def image_callback(self, msg):
    #     """存储最新的图像"""
    #     try:
    #         # 确保图像是BGR格式，适合OpenCV处理
    #         cv_image = self.bridge.imgmsg_to_cv2(msg, "bgr8") 
    #         with self.image_lock:
    #             self.latest_image = cv_image.copy() # 使用copy确保线程安全
    #             success = cv2.imwrite(self.img_save_path , self.latest_image)
    #         if not success:
    #             rospy.logwarn("Failed to save image: person.jpg")
    #     except Exception as e:
    #         rospy.logerr(f"Image conversion error: {str(e)}")

    def image_callback(self, msg):
        """存储最新的图像"""
        try:
            # 确保图像是BGR格式，适合OpenCV处理
            cv_image = self.bridge.imgmsg_to_cv2(msg, "bgr8") 
            with self.image_lock:
                self.latest_image = cv_image.copy() # 使用copy确保线程安全
                
        except Exception as e:
            rospy.logerr(f"Image conversion error: {str(e)}")

    def take_photo1(self):
        if self.latest_image is not None:
        
            # 保存图像
            cv2.imwrite(self.img_save_path1, self.latest_image)
            rospy.loginfo(f"已保存照片: {self.img_save_path1}")
        else:
            rospy.logwarn("无可用图像，拍照失败！")
        
    def take_photo2(self):
        if self.latest_image is not None:
        
            # 保存图像
            cv2.imwrite(self.img_save_path2, self.latest_image)
            rospy.loginfo(f"已保存照片: {self.img_save_path2}")
        else:
            rospy.logwarn("无可用图像，拍照失败！")

    def letterbox(self, img: np.ndarray, new_shape: tuple = (512, 512), color=(114, 114, 114)):
        shape = img.shape[:2]
        r = min(new_shape[0] / shape[0], new_shape[1] / shape[1])
        new_unpad = (int(round(shape[1] * r)), int(round(shape[0] * r)))
        dw = new_shape[1] - new_unpad[0]
        dh = new_shape[0] - new_unpad[1]
        dw /= 2
        dh /= 2
        if shape[::-1] != new_unpad:
            img = cv2.resize(img, new_unpad, interpolation=cv2.INTER_LINEAR)
        top, bottom = int(round(dh - 0.1)), int(round(dh + 0.1))
        left, right = int(round(dw - 0.1)), int(round(dw + 0.1))
        img = cv2.copyMakeBorder(img, top, bottom, left, right,
                                 cv2.BORDER_CONSTANT, value=color)
        ratio = r
        pad = (top, left)
        return img, pad, ratio

    def preprocess(self, frame: np.ndarray):
        self.img_height, self.img_width = frame.shape[:2]
        img_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        img_lb, pad, ratio = self.letterbox(img_rgb, (self.input_width, self.input_height))
        self.pad = pad
        self.ratio = ratio
        image_data = np.array(img_lb) / 255.0
        image_data = np.transpose(image_data, (2, 0, 1))
        return np.expand_dims(image_data, axis=0).astype(np.float32)

    def detect_all(self, image_path: str) -> List[DetectionResult]:
        frame = cv2.imread(image_path)
        
        if frame is None:
            raise FileNotFoundError(f"无法加载图像: {image_path}")
        self.source_frame = frame.copy()    # 保存原始图，后面可视化用
        img_data = self.preprocess(frame)
        outputs = self.session.run(None, {self.session.get_inputs()[0].name: img_data})
        return self.postprocess_all(outputs)

    def postprocess_all(self, output) -> List[DetectionResult]:
        outputs = np.transpose(np.squeeze(output[0]))
        rows = outputs.shape[0]
        raw_results: List[DetectionResult] = []

        for i in range(rows):
            classes_scores = outputs[i][4:]
            max_score = np.amax(classes_scores)
            if max_score >= self.confidence_thres:
                class_id = int(np.argmax(classes_scores))
                x, y, w, h = outputs[i][0:4]
                # 去除 padding 并映射回原图
                x -= self.pad[1]
                y -= self.pad[0]
                left = int((x - w / 2) / self.ratio)
                top = int((y - h / 2) / self.ratio)
                width = int(w / self.ratio)
                height = int(h / self.ratio)

                raw_results.append(DetectionResult(
                    x=left, y=top, w=width, h=height,
                    label=self.classes[class_id], score=max_score
                ))

        # 执行 NMS 去重
        return self.nms(raw_results, self.iou_thres)

    def nms(self, detections: List[DetectionResult], iou_thres: float) -> List[DetectionResult]:
        detections = sorted(detections, key=lambda x: x.score, reverse=True)
        keep = []
        while detections:
            best = detections.pop(0)
            keep.append(best)
            detections = [d for d in detections if self.iou(best, d) < iou_thres]
        return keep

    def iou(self, box1: DetectionResult, box2: DetectionResult) -> float:
        xi1 = max(box1.x, box2.x)
        yi1 = max(box1.y, box2.y)
        xi2 = min(box1.x + box1.w, box2.x + box2.w)
        yi2 = min(box1.y + box1.h, box2.y + box2.h)
        inter_area = max(0, xi2 - xi1) * max(0, yi2 - yi1)
        box1_area = box1.w * box1.h
        box2_area = box2.w * box2.h
        union_area = box1_area + box2_area - inter_area
        return inter_area / union_area if union_area > 0 else 0.0

    def visualize_on_source(self, results: List[DetectionResult], image_path: str):
        """
        在原图上绘制所有检测结果并保存至 ./output 文件夹（自动适配 Windows/Linux）
        :param results: 检测结果列表
        :param image_path: 输入图片路径（用于自动命名输出文件）
        """
        if not results:
            print("[INFO] 未检测到社区人员")
            return

        vis_frame = self.source_frame.copy()

        # 给不同目标类别随机配置颜色（如未提前定义 color_palette）
        if not hasattr(self, "color_palette"):
            import numpy as np
            self.color_palette = np.random.uniform(0, 255, size=(len(self.classes), 3))

        for res in results:
            class_index = self.classes.index(res.label)
            color = tuple(int(c) for c in self.color_palette[class_index])
            # 绘制矩形框
            cv2.rectangle(vis_frame,
                          (res.x, res.y),
                          (res.x + res.w, res.y + res.h),
                          color, 2)
            # 绘制标签
            cv2.putText(vis_frame,
                        f"{res.label} {res.score:.2f}",
                        (res.x, max(0, res.y - 5)),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.6, color, 2)

        # ===== 自动生成输出路径 =====
        # base_dir = os.path.dirname(os.path.abspath(__file__))  # 当前脚本目录
        # output_dir = os.path.join(base_dir, "output")
        # os.makedirs(output_dir, exist_ok=True)

        # 生成以输入图像名为基础的输出文件名
        # image_name = os.path.basename(image_path)
        # name, ext = os.path.splitext(image_name)
        # save_path = os.path.join(output_dir, f"{name}_detected{ext}")
        # save_path = os.path.normpath(save_path)  # 统一路径格式

        # ===== 保存 =====
        success = cv2.imwrite(image_path, vis_frame)
        if success:
            rospy.loginfo(f"[INFO] 可视化结果已保存到: {image_path}")
        else:
            rospy.loginfo(f"[ERROR] 保存失败，请检查路径或写入权限: {image_path}")

    def handle_recognition(self, req):
        """处理服务请求"""
        if req.detect_flag==1:
            rospy.loginfo("I GOT Request:%d,Starting person detection",req.detect_flag)
            # 获取最新图像
            with self.image_lock:
                current_image = self.latest_image.copy() # 再次copy，确保在处理过程中不被订阅回调修改
                self.take_photo1()
                
            # 检查是否有可用图像
            if current_image is None:
                rospy.logwarn("No image available for recognition")
                return detectResponse("ERROR: No image available")
                
            # 1. 执行识别
            result = self.detect_all(self.img_save_path1)
            
            if result:
                # 2. 在图像上绘制识别信息,并且保存
                self.visualize_on_source(result,self.detected_img_save_path1)
                rospy.loginfo(f"Successfully saved image to: {self.detected_img_save_path1}")
                    
                # 4. 返回给服务调用方
                if result != 'N/A':
                    total_people = len(result)
                    community_count = sum(1 for r in result if r.label == "community")
                    uncommunity_count = sum(1 for r in result if r.label == "uncommunity")
                    info_str = f"{total_people},{community_count},{uncommunity_count}"
                    rospy.loginfo(f"图中共有 {total_people} 人，其中社区人员 {community_count} 人，非社区人员 {uncommunity_count} 人")
                    return detectResponse(info_str)
                else:
                    return detectResponse("RECOGNITION_FAILED")

            else:
                rospy.logwarn("Plate recognition failed (API returned no valid result)")
                return detectResponse("RECOGNITION_FAILED")
            
        if req.detect_flag==2:
            rospy.loginfo("I GOT Request:%d,Starting person detection",req.detect_flag)
            # 获取最新图像
            with self.image_lock:
                current_image = self.latest_image.copy() # 再次copy，确保在处理过程中不被订阅回调修改
                self.take_photo2()
                
            # 检查是否有可用图像
            if current_image is None:
                rospy.logwarn("No image available for recognition")
                return detectResponse("ERROR: No image available")
                
            # 1. 执行识别
            result = self.detect_all(self.img_save_path2)
            
            if result:
                # 2. 在图像上绘制识别信息,并且保存
                self.visualize_on_source(result,self.detected_img_save_path2)
                rospy.loginfo(f"Successfully saved image to: {self.detected_img_save_path2}")
                    
                # 4. 返回给服务调用方
                if result != 'N/A':
                    total_people = len(result)
                    community_count = sum(1 for r in result if r.label == "community")
                    uncommunity_count = sum(1 for r in result if r.label == "uncommunity")
                    info_str = f"{total_people},{community_count},{uncommunity_count}"
                    rospy.loginfo(f"图中共有 {total_people} 人，其中社区人员 {community_count} 人，非社区人员 {uncommunity_count} 人")
                    return detectResponse(info_str)
                else:
                    return detectResponse("RECOGNITION_FAILED")

            
# 定义类别标签
classes = ["community", "uncommunity"]

if __name__ == '__main__':
    try:
        detector = MultipleDetection("/home/zcy/simulation_ws/src/robot_navigation/models/yolo11s-community.onnx",
                   confidence_thres=0.8,
                   iou_thres=0.5,
                   class_names=classes)
        rospy.spin()
    except rospy.ROSInterruptException:
        pass
    
