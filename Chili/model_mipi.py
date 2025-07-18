import cv2
import numpy as np
from scipy.special import softmax
# from scipy.special import expit as sigmoid
from hobot_dnn import pyeasy_dnn as dnn  # BSP Python API

from time import time
import argparse
import logging 


# 日志模块配置
# logging configs
logging.basicConfig(
    level = logging.DEBUG,
    format = '[%(name)s] [%(asctime)s.%(msecs)03d] [%(levelname)s] %(message)s',
    datefmt='%H:%M:%S')
logger = logging.getLogger("RDK_YOLO")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model-path', type=str, default='models/yolov5s_672x672_nv12.bin', 
                        help="""Path to BPU Quantized *.bin Model.
                                RDK X3(Module): Bernoulli2.
                                RDK Ultra: Bayes.
                                RDK X5(Module): Bayes-e.
                                RDK S100: Nash-e.
                                RDK S100P: Nash-m.""") 
    parser.add_argument('--test-img', type=str, default='../../../resource/assets/bus.jpg', help='Path to Load Test Image.')
    parser.add_argument('--img-save-path', type=str, default='jupyter_result.jpg', help='Path to Load Test Image.')
    parser.add_argument('--classes-num', type=int, default=80, help='Classes Num to Detect.')
    parser.add_argument('--iou-thres', type=float, default=0.45, help='IoU threshold.')
    parser.add_argument('--conf-thres', type=float, default=0.25, help='confidence threshold.')
    parser.add_argument('--anchors', type=lambda s: list(map(int, s.split(','))), 
                        default=[10, 13, 16, 30, 33, 23, 30, 61, 62, 45, 59, 119, 116, 90, 156, 198, 373, 326],
                        help='--anchors 10,13,16,30,33,23,30,61,62,45,59,119,116,90,156,198,373,326')
    parser.add_argument('--strides', type=lambda s: list(map(int, s.split(','))), 
                        default=[8, 16, 32],
                        help='--strides 8,16,32')
    opt = parser.parse_args()
    logger.info(opt)

    # 实例化
    model = YOLOv5_Detect(opt.model_path, opt.conf_thres, opt.iou_thres, opt.classes_num, opt.anchors, opt.strides)
    # 读图
    img = cv2.imread(opt.test_img)
    # 准备输入数据
    input_tensor = model.bgr2nv12(img)
    # 推理
    outputs = model.c2numpy(model.forward(input_tensor))
    # 后处理
    ids, scores, bboxes = model.postProcess(outputs)
    # 渲染
    logger.info("\033[1;32m" + "Draw Results: " + "\033[0m")
    for class_id, score, bbox in zip(ids, scores, bboxes):
        x1, y1, x2, y2 = bbox
        logger.info("(%d, %d, %d, %d) -> %s: %.2f"%(x1,y1,x2,y2, coco_names[class_id], score))
        draw_detection(img, (x1, y1, x2, y2), score, class_id)
    # 保存结果
    cv2.imwrite(opt.img_save_path, img)
    logger.info("\033[1;32m" + f"saved in path: \"./{opt.img_save_path}\"" + "\033[0m")

class BaseModel:
    def __init__(
        self,
        model_file: str
        ) -> None:
        # 加载BPU的bin模型, 打印相关参数
        # Load the quantized *.bin model and print its parameters
        try:
            begin_time = time()
            self.quantize_model = dnn.load(model_file)
            logger.debug("\033[1;31m" + "Load D-Robotics Quantize model time = %.2f ms"%(1000*(time() - begin_time)) + "\033[0m")
        except Exception as e:
            logger.error("? Failed to load model file: %s"%(model_file))
            logger.error("You can download the model file from the following docs: ./models/download.md") 
            logger.error(e)
            exit(1)

        logger.info("\033[1;32m" + "-> input tensors" + "\033[0m")
        for i, quantize_input in enumerate(self.quantize_model[0].inputs):
            logger.info(f"intput[{i}], name={quantize_input.name}, type={quantize_input.properties.dtype}, shape={quantize_input.properties.shape}")

        logger.info("\033[1;32m" + "-> output tensors" + "\033[0m")
        for i, quantize_input in enumerate(self.quantize_model[0].outputs):
            logger.info(f"output[{i}], name={quantize_input.name}, type={quantize_input.properties.dtype}, shape={quantize_input.properties.shape}")

        self.model_input_height, self.model_input_weight = self.quantize_model[0].inputs[0].properties.shape[2:4]

    def resizer(self, img: np.ndarray)->np.ndarray:
        img_h, img_w = img.shape[0:2]
        self.y_scale, self.x_scale = img_h/self.model_input_height, img_w/self.model_input_weight
        return cv2.resize(img, (self.model_input_height, self.model_input_weight), interpolation=cv2.INTER_NEAREST) # 利用resize重新开辟内存
    
    def preprocess(self, img: np.ndarray)->np.array:
        """
        Preprocesses an input image to prepare it for model inference.

        Args:
            img (np.ndarray): The input image in BGR format as a NumPy array.

        Returns:
            np.array: The preprocessed image tensor in NCHW format ready for model input.

        Procedure:
            1. Resizes the image to a specified dimension (`input_image_size`) using nearest neighbor interpolation.
            2. Converts the image color space from BGR to RGB.
            3. Transposes the dimensions of the image tensor to channel-first order (CHW).
            4. Adds a batch dimension, thus conforming to the NCHW format expected by many models.
            Note: Normalization to [0, 1] is assumed to be handled elsewhere based on configuration.
        """
        begin_time = time()

        input_tensor = self.resizer(img)
        input_tensor = cv2.cvtColor(input_tensor, cv2.COLOR_BGR2RGB)
        # input_tensor = np.array(input_tensor) / 255.0  # yaml文件中已经配置前处理
        input_tensor = np.transpose(input_tensor, (2, 0, 1))
        input_tensor = np.expand_dims(input_tensor, axis=0).astype(np.uint8)  # NCHW

        logger.debug("\033[1;31m" + f"pre process time = {1000*(time() - begin_time):.2f} ms" + "\033[0m")
        return input_tensor

    def bgr2nv12(self, bgr_img: np.ndarray) -> np.ndarray:
        """
        Convert a BGR image to the NV12 format.

        NV12 is a common video encoding format where the Y component (luminance) is full resolution,
        and the UV components (chrominance) are half-resolution and interleaved. This function first
        converts the BGR image to YUV 4:2:0 planar format, then rearranges the UV components to fit
        the NV12 format.

        Parameters:
        bgr_img (np.ndarray): The input BGR image array.

        Returns:
        np.ndarray: The converted NV12 format image array.
        """
        begin_time = time()
        bgr_img = self.resizer(bgr_img)
        height, width = bgr_img.shape[0], bgr_img.shape[1]
        area = height * width
        yuv420p = cv2.cvtColor(bgr_img, cv2.COLOR_BGR2YUV_I420).reshape((area * 3 // 2,))
        y = yuv420p[:area]
        uv_planar = yuv420p[area:].reshape((2, area // 4))
        uv_packed = uv_planar.transpose((1, 0)).reshape((area // 2,))
        nv12 = np.zeros_like(yuv420p)
        nv12[:height * width] = y
        nv12[height * width:] = uv_packed

        logger.debug("\033[1;31m" + f"bgr8 to nv12 time = {1000*(time() - begin_time):.2f} ms" + "\033[0m")
        return nv12


    def forward(self, input_tensor: np.array) -> list[dnn.pyDNNTensor]:
        begin_time = time()
        quantize_outputs = self.quantize_model[0].forward(input_tensor)
        logger.debug("\033[1;31m" + f"forward time = {1000*(time() - begin_time):.2f} ms" + "\033[0m")
        return quantize_outputs


    def c2numpy(self, outputs) -> list[np.array]:
        begin_time = time()
        outputs = [dnnTensor.buffer for dnnTensor in outputs]
        logger.debug("\033[1;31m" + f"c to numpy time = {1000*(time() - begin_time):.2f} ms" + "\033[0m")
        return outputs

class YOLOv5_Detect(BaseModel):
    def __init__(self, 
                model_file: str, 
                conf: float, 
                iou: float,
                nc: int,
                anchors: list,
                strides: list
                ):
        super().__init__(model_file)
        # 配置项目
        self.conf = conf
        self.iou = iou
        self.nc = 80
        self.strides = np.array(strides) 
        input_h, input_w = self.model_input_height, self.model_input_weight

        # strides的grid网格, 只需要生成一次
        s_grid = np.stack([np.tile(np.linspace(0.5, input_w//strides[0] - 0.5, input_w//strides[0]), reps=input_h//strides[0]), 
                            np.repeat(np.arange(0.5, input_h//strides[0] + 0.5, 1), input_w//strides[0])], axis=0).transpose(1,0)
        self.s_grid = np.hstack([s_grid, s_grid, s_grid]).reshape(-1, 2)

        m_grid = np.stack([np.tile(np.linspace(0.5, input_w//strides[1] - 0.5, input_w//strides[1]), reps=input_h//strides[1]), 
                            np.repeat(np.arange(0.5, input_h//strides[1] + 0.5, 1), input_w//strides[1])], axis=0).transpose(1,0)
        self.m_grid = np.hstack([m_grid, m_grid, m_grid]).reshape(-1, 2)

        l_grid = np.stack([np.tile(np.linspace(0.5, input_w//strides[2] - 0.5, input_w//strides[2]), reps=input_h//strides[2]), 
                            np.repeat(np.arange(0.5, input_h//strides[2] + 0.5, 1), input_w//strides[2])], axis=0).transpose(1,0)
        self.l_grid = np.hstack([l_grid, l_grid, l_grid]).reshape(-1, 2)

        logger.info(f"{self.s_grid.shape = }  {self.m_grid.shape = }  {self.l_grid.shape = }")

        # 用于广播的anchors, 只需要生成一次
        anchors = np.array(anchors).reshape(3, -1)
        self.s_anchors = np.tile(anchors[0], input_w//strides[0] * input_h//strides[0]).reshape(-1, 2)
        self.m_anchors = np.tile(anchors[1], input_w//strides[1] * input_h//strides[1]).reshape(-1, 2)
        self.l_anchors = np.tile(anchors[2], input_w//strides[2] * input_h//strides[2]).reshape(-1, 2)

        logger.info(f"{self.s_anchors.shape = }  {self.m_anchors.shape = }  {self.l_anchors.shape = }")


    def postProcess(self, outputs: list[np.ndarray]) -> tuple[list]:
        begin_time = time()
        # reshape
        s_pred = outputs[0].reshape([-1, (5 + self.nc)])
        m_pred = outputs[1].reshape([-1, (5 + self.nc)])
        l_pred = outputs[2].reshape([-1, (5 + self.nc)])

        # classify: 利用numpy向量化操作完成阈值筛选 (优化版 2.0)
        s_raw_max_scores = np.max(s_pred[:, 5:], axis=1)
        s_max_scores = 1 / ((1 + np.exp(-s_pred[:, 4]))*(1 + np.exp(-s_raw_max_scores)))
        # s_max_scores = sigmoid(s_pred[:, 4])*sigmoid(s_pred[:, 4])
        s_valid_indices = np.flatnonzero(s_max_scores >= self.conf)
        s_ids = np.argmax(s_pred[s_valid_indices, 5:], axis=1)
        s_scores = s_max_scores[s_valid_indices]

        m_raw_max_scores = np.max(m_pred[:, 5:], axis=1)
        m_max_scores = 1 / ((1 + np.exp(-m_pred[:, 4]))*(1 + np.exp(-m_raw_max_scores)))
        # m_max_scores = sigmoid(m_pred[:, 4])*sigmoid(m_pred[:, 4])
        m_valid_indices = np.flatnonzero(m_max_scores >= self.conf)
        m_ids = np.argmax(m_pred[m_valid_indices, 5:], axis=1)
        m_scores = m_max_scores[m_valid_indices]

        l_raw_max_scores = np.max(l_pred[:, 5:], axis=1)
        l_max_scores = 1 / ((1 + np.exp(-l_pred[:, 4]))*(1 + np.exp(-l_raw_max_scores)))
        # l_max_scores = sigmoid(l_pred[:, 4])*sigmoid(l_pred[:, 4])
        l_valid_indices = np.flatnonzero(l_max_scores >= self.conf)
        l_ids = np.argmax(l_pred[l_valid_indices, 5:], axis=1)
        l_scores = l_max_scores[l_valid_indices]

        # 特征解码
        s_dxyhw = 1 / (1 + np.exp(-s_pred[s_valid_indices, :4]))
        # s_dxyhw = sigmoid(s_pred[s_valid_indices, :4])
        s_xy = (s_dxyhw[:, 0:2] * 2.0 + self.s_grid[s_valid_indices,:] - 1.0) * self.strides[0]
        s_wh = (s_dxyhw[:, 2:4] * 2.0) ** 2 * self.s_anchors[s_valid_indices, :]
        s_xyxy = np.concatenate([s_xy - s_wh * 0.5, s_xy + s_wh * 0.5], axis=-1)

        m_dxyhw = 1 / (1 + np.exp(-m_pred[m_valid_indices, :4]))
        # m_dxyhw = sigmoid(m_pred[m_valid_indices, :4])
        m_xy = (m_dxyhw[:, 0:2] * 2.0 + self.m_grid[m_valid_indices,:] - 1.0) * self.strides[1]
        m_wh = (m_dxyhw[:, 2:4] * 2.0) ** 2 * self.m_anchors[m_valid_indices, :]
        m_xyxy = np.concatenate([m_xy - m_wh * 0.5, m_xy + m_wh * 0.5], axis=-1)

        l_dxyhw = 1 / (1 + np.exp(-l_pred[l_valid_indices, :4]))
        # l_dxyhw = sigmoid(l_pred[l_valid_indices, :4])
        l_xy = (l_dxyhw[:, 0:2] * 2.0 + self.l_grid[l_valid_indices,:] - 1.0) * self.strides[2]
        l_wh = (l_dxyhw[:, 2:4] * 2.0) ** 2 * self.l_anchors[l_valid_indices, :]
        l_xyxy = np.concatenate([l_xy - l_wh * 0.5, l_xy + l_wh * 0.5], axis=-1)

        # 大中小特征层阈值筛选结果拼接
        xyxy = np.concatenate((s_xyxy, m_xyxy, l_xyxy), axis=0)
        scores = np.concatenate((s_scores, m_scores, l_scores), axis=0)
        ids = np.concatenate((s_ids, m_ids, l_ids), axis=0)

        # nms
        indices = cv2.dnn.NMSBoxes(xyxy, scores, self.conf, self.iou)

        # 还原到原始的img尺度
        bboxes = (xyxy[indices] * np.array([self.x_scale, self.y_scale, self.x_scale, self.y_scale])).astype(np.int32)

        logger.debug("\033[1;31m" + f"Post Process time = {1000*(time() - begin_time):.2f} ms" + "\033[0m")

        return ids[indices], scores[indices], bboxes


coco_names = [
    "person", "bicycle", "car", "motorcycle", "airplane", "bus", "train", "truck", "boat", "traffic light", 
    "fire hydrant", "stop sign", "parking meter", "bench", "bird", "cat", "dog", "horse", "sheep", "cow", 
    "elephant", "bear", "zebra", "giraffe", "backpack", "umbrella", "handbag", "tie", "suitcase", "frisbee", 
    "skis", "snowboard", "sports ball", "kite", "baseball bat", "baseball glove", "skateboard", "surfboard", "tennis racket", "bottle", 
    "wine glass", "cup", "fork", "knife", "spoon", "bowl", "banana", "apple", "sandwich", "orange", 
    "broccoli", "carrot", "hot dog", "pizza", "donut", "cake", "chair", "couch", "potted plant", "bed", 
    "dining table", "toilet", "tv", "laptop", "mouse", "remote", "keyboard", "cell phone", "microwave", "oven", 
    "toaster", "sink", "refrigerator", "book", "clock", "vase", "scissors", "teddy bear", "hair drier", "toothbrush"
    ]

rdk_colors = [
    (56, 56, 255), (151, 157, 255), (31, 112, 255), (29, 178, 255),(49, 210, 207), (10, 249, 72), (23, 204, 146), (134, 219, 61),
    (52, 147, 26), (187, 212, 0), (168, 153, 44), (255, 194, 0),(147, 69, 52), (255, 115, 100), (236, 24, 0), (255, 56, 132),
    (133, 0, 82), (255, 56, 203), (200, 149, 255), (199, 55, 255)]

def draw_detection(img: np.array, 
                   bbox,#: tuple[int, int, int, int],
                   score: float, 
                   class_id: int) -> None:
    """
    Draws a detection bounding box and label on the image.

    Parameters:
        img (np.array): The input image.
        bbox (tuple[int, int, int, int]): A tuple containing the bounding box coordinates (x1, y1, x2, y2).
        score (float): The detection score of the object.
        class_id (int): The class ID of the detected object.
    """
    x1, y1, x2, y2 = bbox
    color = rdk_colors[class_id%20]
    cv2.rectangle(img, (x1, y1), (x2, y2), color, 2)
    label = f"{coco_names[class_id]}: {score:.2f}"
    (label_width, label_height), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
    label_x, label_y = x1, y1 - 10 if y1 - 10 > label_height else y1 + 10
    cv2.rectangle(
        img, (label_x, label_y - label_height), (label_x + label_width, label_y + label_height), color, cv2.FILLED
    )
    cv2.putText(img, label, (label_x, label_y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1, cv2.LINE_AA)

if __name__ == "__main__":
    main()
