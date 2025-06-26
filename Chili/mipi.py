import sys
import signal
import os
import numpy as np
import cv2
import colorsys
from time import time, sleep
import multiprocessing
from threading import BoundedSemaphore
import ctypes
import json
from hobot_vio import libsrcampy as srcampy
from hobot_dnn import pyeasy_dnn as dnn
import threading
os.environ['DISPLAY'] = ':0'
os.environ['EGL_PLATFORM'] = 'surfaceless'

# 传感器参数
sensor_width = 1920
sensor_height = 1080

# 全局变量
image_counter = None
is_stop = False
disp = None
cam = None

def signal_handler(signal, frame):
    global is_stop, cam, disp
    print("Stopping!\n")
    is_stop = True
    if cam is not None:
        cam.close_cam()
    if disp is not None:
        disp.close()
    sys.exit(0)

def get_display_res():
    disp_w_small = 1920
    disp_h_small = 1080
    disp = srcampy.Display()
    resolution_list = disp.get_display_res()
    
    if (sensor_width, sensor_height) in resolution_list:
        print(f"Resolution {sensor_width}x{sensor_height} exists in the list.")
        return int(sensor_width), int(sensor_height)
    else:
        print(f"Resolution {sensor_width}x{sensor_height} does not exist in the list.")
        for res in resolution_list:
            if res[0] == 0 and res[1] == 0:
                break
            else:
                disp_w_small = res[0]
                disp_h_small = res[1]
                
            if res[0] <= sensor_width and res[1] <= sensor_height:
                print(f"Resolution {res[0]}x{res[1]}.")
                return int(res[0]), int(res[1])
    
    disp.close()
    return disp_w_small, disp_h_small

disp_w, disp_h = get_display_res()

# YOLOv5模型类别名称 - 根据您的实际模型修改
def get_classes():
    return np.array([
        "Chili"
    ])

def get_hw(pro):
    if pro.layout == "NCHW":
        return pro.shape[2], pro.shape[3]
    else:
        return pro.shape[1], pro.shape[2]

def print_properties(pro):
    print("tensor type:", pro.tensor_type)
    print("data type:", pro.dtype)
    print("layout:", pro.layout)
    print("shape:", pro.shape)

class ParallelExector(object):
    def __init__(self, counter, parallel_num=4):
        global image_counter
        image_counter = counter
        self.parallel_num = parallel_num
        if parallel_num != 1:
            self._pool = multiprocessing.Pool(processes=self.parallel_num,
                                            maxtasksperchild=5)
            self.workers = BoundedSemaphore(self.parallel_num)

    def infer(self, output):
        if self.parallel_num == 1:
            run(output)
        else:
            self.workers.acquire()
            self._pool.apply_async(func=run,
                                  args=(output,),
                                  callback=self.task_done,
                                  error_callback=print)

    def task_done(self, *args, **kwargs):
        self.workers.release()

    def close(self):
        if hasattr(self, "_pool"):
            self._pool.close()
            self._pool.join()

def limit_display_cord(coor):
    coor[0] = max(min(disp_w, coor[0]), 0)
    coor[1] = max(min(disp_h, coor[1]), 2)
    coor[2] = max(min(disp_w, coor[2]), 0)
    coor[3] = max(min(disp_h, coor[3]), 0)
    return coor

def scale_bbox(bbox, input_w, input_h, output_w, output_h):
    scale_x = output_w / input_w
    scale_y = output_h / input_h

    x1 = int(bbox[0] * scale_x)
    y1 = int(bbox[1] * scale_y)
    x2 = int(bbox[2] * scale_x)
    y2 = int(bbox[3] * scale_y)

    return [x1, y1, x2, y2]

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def xywh2xyxy(x):
    # Convert nx4 boxes from [x, y, w, h] to [x1, y1, x2, y2] where xy1=top-left, xy2=bottom-right
    y = np.copy(x)
    y[:, 0] = x[:, 0] - x[:, 2] / 2  # top left x
    y[:, 1] = x[:, 1] - x[:, 3] / 2  # top left y
    y[:, 2] = x[:, 0] + x[:, 2] / 2  # bottom right x
    y[:, 3] = x[:, 1] + x[:, 3] / 2  # bottom right y
    return y

class YOLOv5PostProcess:
    def __init__(self, img_size=640, num_classes=80, conf_thres=0.5, iou_thres=0.6):
        self.img_size = img_size
        self.num_classes = num_classes
        self.conf_thres = conf_thres
        self.iou_thres = iou_thres
        
        # YOLOv5 anchors (P3, P4, P5) - 根据您的模型调整
        self.anchors = np.array([
            [[10, 13], [16, 30], [33, 23]],     # P3/8
            [[30, 61], [62, 45], [59, 119]],    # P4/16
            [[116, 90], [156, 198], [373, 326]] # P5/32
        ], dtype=np.float32)
        
        self.stride = np.array([8., 16., 32.])
        self.grid = [np.zeros(1)] * 3  # init grid
        self.anchor_grid = [np.zeros(1)] * 3  # init anchor grid
        
    def make_grid(self, nx=20, ny=20, i=0):
        yv, xv = np.meshgrid(np.arange(ny), np.arange(nx))
        grid = np.stack((xv, yv), 2)
        anchor_grid = (self.anchors[i] * self.stride[i]).reshape(1, 3, 1, 1, 2)
        return grid, anchor_grid
    
    def __call__(self, outputs):
        """
        YOLOv5后处理
        outputs: 模型输出的三个特征图 [small, medium, large]
        """
        z = []  # 存储所有检测结果
        for i, output in enumerate(outputs):
            # output shape: (bs, 3, ny, nx, no) where no = 5 + nc
            bs, _, ny, nx, _ = output.shape
            if self.grid[i].shape[2:4] != (ny, nx):
                self.grid[i], self.anchor_grid[i] = self.make_grid(nx, ny, i)
            
            # 转换为numpy array
            output = output.squeeze()  # (3, ny, nx, 85)
            
            # 应用sigmoid到置信度和类别概率
            output[..., 4:] = sigmoid(output[..., 4:])
            
            # 调整预测框坐标
            y = output.copy()
            y[..., 0:2] = (y[..., 0:2] * 2 - 0.5 + self.grid[i]) * self.stride[i]  # xy
            y[..., 2:4] = (y[..., 2:4] * 2) ** 2 * self.anchor_grid[i]  # wh
            
            # 转换为(x1, y1, x2, y2)格式
            y = y.reshape(3 * ny * nx, -1)
            box = xywh2xyxy(y[:, :4])
            
            # 添加类别置信度
            obj_conf = y[:, 4:5]
            cls_conf = y[:, 5:] * obj_conf
            box_cls_conf = np.concatenate((box, cls_conf), 1)
            
            # 过滤低置信度检测
            box_cls_conf = box_cls_conf[box_cls_conf[:, 4] > self.conf_thres]
            z.append(box_cls_conf)
        
        # 合并所有尺度的检测结果
        if len(z) > 0:
            z = np.concatenate(z, 0)
            
            # 非极大值抑制(NMS)
            boxes = z[:, :4]
            scores = z[:, 4]
            classes = np.argmax(z[:, 5:], axis=1)
            
            # 使用OpenCV的NMS
            indices = cv2.dnn.NMSBoxes(boxes.tolist(), scores.tolist(), 
                                      self.conf_thres, self.iou_thres)
            
            if len(indices) > 0:
                indices = indices.flatten()
                return boxes[indices], scores[indices], classes[indices]
        
        return np.zeros((0, 4)), np.zeros(0), np.zeros(0)

def run(outputs):
    global image_counter, disp
    
    try:
        # 1. 处理模型输出
        output_np = []
        for i, output in enumerate(outputs):
            # 反量化处理
            if hasattr(models[0].outputs[i].properties, 'scale_data'):
                scale = models[0].outputs[i].properties.scale_data[0]
                zero_point = models[0].outputs[i].properties.zero_point_data[0] if hasattr(
                    models[0].outputs[i].properties, 'zero_point_data') else 0
                output = (output.astype(np.float32) - zero_point) * scale
            
            # 动态形状调整
            if output.size == 7200:  # 小尺寸输出
                reshaped = output.reshape(1, 3, 20, 20, 6)
            elif output.size == 28800:  # 中尺寸输出
                reshaped = output.reshape(1, 3, 40, 40, 6)
            elif output.size == 115200:  # 大尺寸输出
                reshaped = output.reshape(1, 3, 80, 80, 6)
            else:
                raise ValueError(f"Unexpected output size: {output.size}")
            
            output_np.append(reshaped)
        
        # 2. 执行后处理
        postprocessor = YOLOv5PostProcess(img_size=640, num_classes=len(get_classes()))
        boxes, scores, classes = postprocessor(output_np)
        
        # 3. 显示结果
        classes_list = get_classes()
        for i, (box, score, cls_id) in enumerate(zip(boxes, scores, classes)):
            box = scale_bbox(box, 640, 640, disp_w, disp_h)
            coor = limit_display_cord(box)
            coor = [round(i) for i in coor]
            
            cls_name = classes_list[cls_id]
            bbox_string = f"{cls_name}: {score:.2f}".encode('gb2312')
            box_color = colors[cls_id % len(colors)]
            color_ARGB = 0xFF000000 | (box_color[0] << 16) | (box_color[1] << 8) | box_color[2]
            
            disp.set_graph_rect(*coor, 3, int(i==0), color_ARGB)
            disp.set_graph_word(coor[0], coor[1]-2, bbox_string, 3, int(i==0), color_ARGB)
            
        # 4. 更新帧率计数
        with image_counter.get_lock():
            image_counter.value += 1
            if image_counter.value % 100 == 0:
                print(f"FPS: {100/(time()-start_time):.2f}")
                start_time = time()
                
    except Exception as e:
        print(f"Error in processing: {str(e)}")
        import traceback
        traceback.print_exc()

if __name__ == '__main__':
    signal.signal(signal.SIGINT, signal_handler)

    # 加载YOLOv5模型
    models = dnn.load('Chili.bin')  # 替换为您的模型路径
    print("=== Model Output Properties ===")

    print("--- model input properties ---")
    print_properties(models[0].inputs[0].properties)
    print("--- model output properties ---")
    for output in models[0].outputs:
        print_properties(output.properties)
    for i, output in enumerate(models[0].outputs):
    print(f"Output {i}:")
    print(f"Shape: {output.properties.shape}")
    print(f"Data type: {output.properties.dtype}")
    print(f"Quantization type: {getattr(output.properties, 'quantiType', 'N/A')}")
    print(f"Scale data: {getattr(output.properties, 'scale_data', 'N/A')}")
    print(f"Scale struct: {getattr(output.properties, 'scale', 'N/A')}")
    print("-" * 40)
    # 初始化摄像头
    cam = srcampy.Camera()
    h, w = get_hw(models[0].inputs[0].properties)
    input_shape = (h, w)
    
    # 打开摄像头 (根据您的摄像头调整参数)
    cam.open_cam(0, -1, -1, [w, disp_w], [h, disp_h], sensor_height, sensor_width)

    # 初始化显示
    disp = srcampy.Display()
    disp.display(0, disp_w, disp_h)
    srcampy.bind(cam, disp)
    disp.display(3, disp_w, disp_h)

    # 设置颜色和类别
    classes = get_classes()
    num_classes = len(classes)
    hsv_tuples = [(1.0 * x / num_classes, 1., 1.) for x in range(num_classes)]
    colors = list(map(lambda x: colorsys.hsv_to_rgb(*x), hsv_tuples))
    colors = list(
        map(lambda x: (int(x[0] * 255), int(x[1] * 255), int(x[2] * 255)),
        colors))

    # FPS计时器
    start_time = time()
    image_counter = multiprocessing.Value("i", 0)

    # 并行执行器
    parallel_exe = ParallelExector(image_counter)

    # 主循环
    while not is_stop:
        # 获取图像 (NV12格式)
        img = cam.get_img(2, 640, 640)  # 640x640 NV12格式
        
        # 转换为numpy数组
        img = np.frombuffer(img, dtype=np.uint8)
        
        # 模型推理
        outputs = models[0].forward(img)
        
        # 准备输出数组
        output_array = []
        for item in outputs:
            output_array.append(item.buffer)
        
        # 执行后处理
        parallel_exe.infer(output_array)

    # 清理资源
    cam.close_cam()
    disp.close()