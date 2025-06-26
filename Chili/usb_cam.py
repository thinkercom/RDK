import cv2
import os
import time

# 创建存储图片的文件夹
if not os.path.exists('imgs'):
    os.makedirs('imgs')

# 打开摄像头
cap = cv2.VideoCapture('/dev/video0', cv2.CAP_V4L2)

if not cap.isOpened():
    print("无法打开摄像头")
    exit()

frame_count = 0
start_time = time.time()

while True:
    ret, frame = cap.read()
    if not ret:
        print("无法接收帧（可能是因为到达了流的末尾或遇到了其他问题）")
        break
    
    # 显示实时画面
    cv2.imshow('Camera Feed', frame)
    
    current_time = time.time()
    elapsed_time = current_time - start_time
    
    # 每隔15秒保存一张图片
    if elapsed_time > 5:
        img_name = "imgs/detect_{}.png".format(frame_count)
        cv2.imwrite(img_name, frame)
        print("{} 已保存".format(img_name))
        frame_count += 1
        start_time = current_time  # 更新开始时间以重新计时
    
    # 如果用户按下 'q' 键则退出循环
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# 当一切完成时，释放捕获器
cap.release()
cv2.destroyAllWindows()
