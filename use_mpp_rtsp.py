import os
import cv2
import sys
import argparse
import time
import numpy as np
import glob

sys.path.append('/mnt/rknn_model_zoo')
sys.path.append('/mnt/rknn_model_zoo/examples/yolo11/python')
from py_utils.coco_utils import COCO_test_helper

# 需要安装 PyGObject: pip install PyGObject
import gi
gi.require_version('Gst', '1.0')
from gi.repository import Gst

Gst.init(None)  # 必须加这句，初始化GStreamer

# 常量定义
OBJ_THRESH = 0.25
NMS_THRESH = 0.45
# 常量配置
IMG_SIZE = (640, 640)
PROCESS_FPS = 25 #12 #4 #2  # 每秒处理帧数
CLASSES = ("person")

coco_id_list = [1]


def optimized_stream_inference(model, args):
    """优化后的流媒体推理函数，兼容H.264和H.265"""
    import os
    os.environ["OPENCV_FFMPEG_CAPTURE_OPTIONS"] = "rtsp_transport;tcp"

    # 支持h264和h265自动切换
    gst_templates = [
        (
            f'rtspsrc location={args.rtsp_url} latency=50 ! '
            'rtph264depay ! h264parse ! mppvideodec ! appsink'
        ),
        (
            f'rtspsrc location={args.rtsp_url} latency=50 ! '
            'rtph265depay ! h265parse ! mppvideodec ! appsink'
        ),
    ]

    cap = None
    for gst_str in gst_templates:
        # cap = cv2.VideoCapture(gst_str, cv2.CAP_GSTREAMER)
        cap = cv2.VideoCapture(args.rtsp_url)
        if cap.isOpened():
            print(f"成功打开流，使用管道: {gst_str.split('!')[1].strip()}")
            break
        else:
            if cap is not None:
                cap.release()
    if cap is None or not cap.isOpened():
        print(f"无法打开RTSP流: {args.rtsp_url}")
        return

    co_helper = COCO_test_helper(enable_letter_box=True)
    frame_interval = 1.0 / PROCESS_FPS
    last_process_time = time.time()
    fps_counter = 0
    last_fps_time = time.time()
    
    # 获取并降低采集分辨率
    orig_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    orig_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, orig_width//2)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, orig_height//2)
    print(f"采集分辨率: {orig_width//2}x{orig_height//2}")

    # 显示窗口设置
    cv2.namedWindow('AI Detection', cv2.WINDOW_NORMAL)
    cv2.resizeWindow('AI Detection', orig_width//4, orig_height//4)

    skip_frames = 1  # 跳1帧处理1帧

    while True:
        # 跳帧处理
        for _ in range(skip_frames):
            cap.grab()
            
        ret, frame = cap.retrieve()
        if not ret:
            print("视频帧丢失，尝试重连...")
            cap.release()
            cap = cv2.VideoCapture(args.rtsp_url)
            time.sleep(1)
            continue

        # 强制时间控制
        current_time = time.time()
        
        # 预处理
        processed_img = co_helper.letter_box(
            frame, 
            new_shape=(IMG_SIZE[1], IMG_SIZE[0]),
            pad_color=(114, 114, 114)
        )
        
        # 推理
        start_infer = time.time()
        outputs = model.run([processed_img])
        infer_time = (time.time() - start_infer) * 1000
        
        # 后处理
        boxes, classes, scores = post_process(outputs)
        
        # 显示处理
        display_frame = frame.copy()  # 必须放在if外部
        if boxes is not None:
            real_boxes = co_helper.get_real_box(boxes)
            for box, score, cl in zip(real_boxes, scores, classes):
                if cl != 0:
                    continue
                x1, y1, x2, y2 = map(int, box)
                cv2.rectangle(display_frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                label = f'{CLASSES[cl]} {score:.2f}'
                cv2.putText(display_frame, label, (x1, y1-10), 
                          cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,255,0), 2)
        
        # 强制刷新显示
        cv2.imshow('AI Detection', display_frame)
        
        # 性能统计
        fps_counter += 1
        if current_time - last_fps_time >= 1.0:
            print(f"[性能] 处理帧率: {fps_counter}fps | 推理耗时: {infer_time:.1f}ms")
            fps_counter = 0
            last_fps_time = current_time
        
        last_process_time = current_time
        
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()


def setup_model(args):
    model_path = args.model_path
    if model_path.endswith('.pt') or model_path.endswith('.torchscript'):
        platform = 'pytorch'
        from py_utils.pytorch_executor import Torch_model_container
        model = Torch_model_container(args.model_path)
    elif model_path.endswith('.rknn'):
        platform = 'rknn'
        from py_utils.rknn_executor import RKNN_model_container 
        # 移除非必要参数
        model = RKNN_model_container(args.model_path, args.target)  # 移除device_id参数
    elif model_path.endswith('onnx'):
        platform = 'onnx'
        from py_utils.onnx_executor import ONNX_model_container
        model = ONNX_model_container(args.model_path)
    else:
        assert False, "{} is not rknn/pytorch/onnx model".format(model_path)
    print('Model-{} is {} model, starting val'.format(model_path, platform))
    return model, platform

def post_process(input_data):
    boxes, scores, classes_conf = [], [], []
    defualt_branch=3
    pair_per_branch = len(input_data)//defualt_branch
    # Python 忽略 score_sum 输出
    for i in range(defualt_branch):
        boxes.append(box_process(input_data[pair_per_branch*i]))
        classes_conf.append(input_data[pair_per_branch*i+1])
        scores.append(np.ones_like(input_data[pair_per_branch*i+1][:,:1,:,:], dtype=np.float32))

    def sp_flatten(_in):
        ch = _in.shape[1]
        _in = _in.transpose(0,2,3,1)
        return _in.reshape(-1, ch)

    boxes = [sp_flatten(_v) for _v in boxes]
    classes_conf = [sp_flatten(_v) for _v in classes_conf]
    scores = [sp_flatten(_v) for _v in scores]

    boxes = np.concatenate(boxes)
    classes_conf = np.concatenate(classes_conf)
    scores = np.concatenate(scores)

    # filter according to threshold
    boxes, classes, scores = filter_boxes(boxes, scores, classes_conf)

    # nms
    nboxes, nclasses, nscores = [], [], []
    for c in set(classes):
        inds = np.where(classes == c)
        b = boxes[inds]
        c = classes[inds]
        s = scores[inds]
        keep = nms_boxes(b, s)

        if len(keep) != 0:
            nboxes.append(b[keep])
            nclasses.append(c[keep])
            nscores.append(s[keep])

    if not nclasses and not nscores:
        return None, None, None

    boxes = np.concatenate(nboxes)
    classes = np.concatenate(nclasses)
    scores = np.concatenate(nscores)

    return boxes, classes, scores


def box_process(position):
    grid_h, grid_w = position.shape[2:4]
    col, row = np.meshgrid(np.arange(0, grid_w), np.arange(0, grid_h))
    col = col.reshape(1, 1, grid_h, grid_w)
    row = row.reshape(1, 1, grid_h, grid_w)
    grid = np.concatenate((col, row), axis=1)
    stride = np.array([IMG_SIZE[1]//grid_h, IMG_SIZE[0]//grid_w]).reshape(1,2,1,1)

    position = dfl(position)
    box_xy  = grid +0.5 -position[:,0:2,:,:]
    box_xy2 = grid +0.5 +position[:,2:4,:,:]
    xyxy = np.concatenate((box_xy*stride, box_xy2*stride), axis=1)

    return xyxy


def draw(image, boxes, scores, classes):
    for box, score, cl in zip(boxes, scores, classes):
        top, left, right, bottom = [int(_b) for _b in box]
        print("%s @ (%d %d %d %d) %.3f" % (CLASSES[cl], top, left, right, bottom, score))
        cv2.rectangle(image, (top, left), (right, bottom), (255, 0, 0), 2)
        cv2.putText(image, '{0} {1:.2f}'.format(CLASSES[cl], score),
                    (top, left - 6), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
        
def dfl(position):
    # Distribution Focal Loss (DFL)
    import torch
    x = torch.tensor(position)
    n,c,h,w = x.shape
    p_num = 4
    mc = c//p_num
    y = x.reshape(n,p_num,mc,h,w)
    y = y.softmax(2)
    acc_metrix = torch.tensor(range(mc)).float().reshape(1,1,mc,1,1)
    y = (y*acc_metrix).sum(2)
    return y.numpy()


def filter_boxes(boxes, box_confidences, box_class_probs):
    """Filter boxes with object threshold.
    """
    box_confidences = box_confidences.reshape(-1)
    candidate, class_num = box_class_probs.shape

    class_max_score = np.max(box_class_probs, axis=-1)
    classes = np.argmax(box_class_probs, axis=-1)

    # _class_pos = np.where(class_max_score* box_confidences >= OBJ_THRESH)
    # 添加类别过滤条件（0对应person）
    _class_pos = np.where((class_max_score * box_confidences >= OBJ_THRESH) & (classes == 0))
    scores = (class_max_score* box_confidences)[_class_pos]

    boxes = boxes[_class_pos]
    classes = classes[_class_pos]

    return boxes, classes, scores

def nms_boxes(boxes, scores):
    """Suppress non-maximal boxes.
    # Returns
        keep: ndarray, index of effective boxes.
    """
    x = boxes[:, 0]
    y = boxes[:, 1]
    w = boxes[:, 2] - boxes[:, 0]
    h = boxes[:, 3] - boxes[:, 1]

    areas = w * h
    order = scores.argsort()[::-1]

    keep = []
    while order.size > 0:
        i = order[0]
        keep.append(i)

        xx1 = np.maximum(x[i], x[order[1:]])
        yy1 = np.maximum(y[i], y[order[1:]])
        xx2 = np.minimum(x[i] + w[i], x[order[1:]] + w[order[1:]])
        yy2 = np.minimum(y[i] + h[i], y[order[1:]] + h[order[1:]])

        w1 = np.maximum(0.0, xx2 - xx1 + 0.00001)
        h1 = np.maximum(0.0, yy2 - yy1 + 0.00001)
        inter = w1 * h1

        ovr = inter / (areas[i] + areas[order[1:]] - inter)
        inds = np.where(ovr <= NMS_THRESH)[0]
        order = order[inds + 1]
    keep = np.array(keep)
    return keep

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='优化版目标检测程序')
    parser.add_argument('--model_path', required=True, help='RKNN模型路径')
    parser.add_argument('--rtsp_url', required=True, help='RTSP流地址')
    parser.add_argument('--target', default='rk3588', help='硬件平台')
    
    args = parser.parse_args()
    model, _ = setup_model(args)

    # # 用GStreamer硬解码采帧
    # pipeline_str = (
    #     f'rtspsrc location={args.rtsp_url} latency=50 drop-on-latency=true ! '
    #     'rtph264depay ! h264parse ! mppvideodec ! videoconvert ! '
    #     'videoscale ! video/x-raw,width=640,height=360,format=BGR ! '
    #     'appsink name=sink sync=false max-buffers=16 drop=true'
    # )
    # pipeline = Gst.parse_launch(pipeline_str)
    # appsink = pipeline.get_by_name('sink')
    # pipeline.set_state(Gst.State.PLAYING)

    # co_helper = COCO_test_helper(enable_letter_box=True)
    # IMG_SIZE = (640, 640)
    # PROCESS_FPS = 25 #12
    # frame_interval = 1.0 / PROCESS_FPS
    # last_process_time = time.time()
    # fps_counter = 0
    # last_fps_time = time.time()

    # start_time = time.time()
    # frame_count = 0
    # while True:
    #     sample = None
    #     while True:
    #         s = appsink.emit('try-pull-sample', 0)
    #         if s is None:
    #             break
    #         sample = s
    #     if sample:
    #         current_time = time.time()  # 修正：加上这一行
    #         buf = sample.get_buffer()
    #         caps = sample.get_caps()
    #         width = caps.get_structure(0).get_value('width')
    #         height = caps.get_structure(0).get_value('height')
    #         arr = np.ndarray(
    #             (height, width, 3),
    #             buffer=buf.extract_dup(0, buf.get_size()),
    #             dtype=np.uint8
    #         )

    #         # 不要加任何 frame_interval 限制
    #         # 只处理最新一帧，丢弃积压帧
    #         # 统计fps
    #         # 预处理
    #         processed_img = co_helper.letter_box(
    #             arr, 
    #             new_shape=(IMG_SIZE[1], IMG_SIZE[0]),
    #             pad_color=(114, 114, 114)
    #         )

    #         # 推理
    #         start_infer = time.time()
    #         outputs = model.run([processed_img])
    #         infer_time = (time.time() - start_infer) * 1000

    #         # 后处理
    #         boxes, classes, scores = post_process(outputs)

    #         # 显示
    #         display_frame = arr.copy()
    #         if boxes is not None:
    #             real_boxes = co_helper.get_real_box(boxes)
    #             for box, score, cl in zip(real_boxes, scores, classes):
    #                 if cl != 0:
    #                     continue
    #                 x1, y1, x2, y2 = map(int, box)
    #                 cv2.rectangle(display_frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
    #                 label = f'{CLASSES[cl]} {score:.2f}'
    #                 cv2.putText(display_frame, label, (x1, y1-10), 
    #                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,255,0), 2)

    #         cv2.imshow('AI Detection', display_frame)

    #         # 性能统计
    #         fps_counter += 1
    #         frame_count += 1
    #         if current_time - last_fps_time >= 1.0:
    #             print(f"[性能] 处理帧率: {fps_counter}fps | 推理耗时: {infer_time:.1f}ms")
    #             print(f"[性能] 实际处理帧率: {frame_count}fps")
    #             fps_counter = 0
    #             frame_count = 0
    #             last_fps_time = current_time

    #         last_process_time = current_time

    #         if cv2.waitKey(1) & 0xFF == ord('q'):
    #             break
    #     else:
    #         time.sleep(0.001)
    # pipeline.set_state(Gst.State.NULL)
    # cv2.destroyAllWindows()
    # model.release()

    total_start = time.time()
    # ...推理和后处理...
    print(f"总耗时: {(time.time()-total_start)*1000:.1f}ms")

    co_helper = COCO_test_helper(enable_letter_box=True)
    IMG_SIZE = (640, 640)
    last_file = ""
    fps_counter = 0
    last_fps_time = time.time()

    while True:
        files = sorted(glob.glob("/tmp/jpgs/rtsp_ffmpeg_*.jpg"))
        if not files:
            time.sleep(0.01)
            continue
        latest = files[-1]
        if latest == last_file:
            time.sleep(0.01)
            continue
        img = cv2.imread(latest)
        last_file = latest
        if img is not None:
            # 预处理
            processed_img = co_helper.letter_box(
                img, 
                new_shape=(IMG_SIZE[1], IMG_SIZE[0]),
                pad_color=(114, 114, 114)
            )
            # 推理
            start_infer = time.time()
            outputs = model.run([processed_img])
            infer_time = (time.time() - start_infer) * 1000
            # 后处理
            boxes, classes, scores = post_process(outputs)
            # 显示
            display_frame = img.copy()
            if boxes is not None:
                real_boxes = co_helper.get_real_box(boxes)
                for box, score, cl in zip(real_boxes, scores, classes):
                    if cl != 0:
                        continue
                    x1, y1, x2, y2 = map(int, box)
                    cv2.rectangle(display_frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                    label = f'{CLASSES[cl]} {score:.2f}'
                    cv2.putText(display_frame, label, (x1, y1-10), 
                              cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,255,0), 2)
            cv2.imshow("AI Detection", display_frame)
            # 性能统计
            fps_counter += 1
            current_time = time.time()
            if current_time - last_fps_time >= 1.0:
                print(f"[性能] 处理帧率: {fps_counter}fps | 推理耗时: {infer_time:.1f}ms---new")
                fps_counter = 0
                last_fps_time = current_time
            # 删除已处理图片，防止磁盘占满
            try:
                os.remove(latest)
            except Exception as e:
                print(f"删除图片失败: {e}")
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
    cv2.destroyAllWindows()