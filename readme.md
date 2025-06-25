

#激活虚拟环境
source venv/bin/activate

#执行脚本
python3 2_rtsp.py --model_path /mnt/rknn_model_zoo/examples/yolo11/python/yolo11n.rknn --rtsp_url rtsp://admin:123456@192.168.30.84:554/profile1 --target rk3588


app.py

#同为rtsp
python3 app.py --model_path yolo11n.rknn --rtsp_url rtsp://admin:123456@192.168.30.84:554/profile1 --target rk3588 --score_thresh 0.6

#跃天rtsp
rtsp://dbtqzh:admin123@192.168.30.66:554/stream/realtime?channel=1&streamtype=1

#查看npu占用率
watch -n 1 "cat /sys/kernel/debug/rknpu/load"


#测试硬解码
gst-launch-1.0 rtspsrc location="rtsp://admin:123456@192.168.30.98:554/profile1" latency=50 ! rtph264depay ! h264parse ! mppvideodec ! autovideosink
gst-launch-1.0 rtspsrc location="rtsp://admin:123456@192.168.30.98:554/profile1" latency=50 ! rtph264depay ! h264parse ! mppvideodec ! videoconvert ! autovideosink
#测试实际帧率
ffprobe -v error -select_streams v:0 -show_entries stream=r_frame_rate -of default=noprint_wrappers=1 "rtsp://admin:123456@192.168.30.98:554/profile1"

# --------------------------------------------使用ffmpeg采集帧 方式-------------------------
#命令采集帧
ffmpeg -rtsp_transport tcp -i "rtsp://admin:123456@192.168.30.98:554/profile1" -vf scale=640:360 -q:v 2 /tmp/rtsp_ffmpeg_%05d.jpg

ffmpeg -rtsp_transport tcp -i "rtsp://dbtqzh:admin123@192.168.30.66:554/stream/realtime?channel=1&streamtype=1" -vf scale=640:360 -q:v 2 /tmp/rtsp_ffmpeg_%05d.jpg

解耦采帧与推理

采用 GStreamer/FFmpeg 命令行工具将 RTSP 流采帧为图片，Python 只负责读取最新图片并推理，避免 Python 端采帧性能瓶颈。
只处理最新帧，丢弃积压帧

Python 端每次只读取最新一张图片，跳过旧帧，保证实时性，避免延迟和堆积。
处理完即删除图片，防止磁盘占满

每处理完一帧后立即删除对应图片，避免硬盘空间被占满。
推理与显示合并主循环

预处理、推理、后处理、显示、性能统计全部在同一个主循环内完成，逻辑清晰。
性能统计

实时统计推理耗时和实际处理帧率，便于分析瓶颈。
模型推理优化

支持更换不同规模的 YOLO11 RKNN 模型（如 yolo11n、yolo11m），可根据推理速度和精度需求灵活调整。
支持调整输入分辨率（如 IMG_SIZE = (640, 640) 或更低）。
异常处理与重连机制

若图片读取失败或帧丢失，自动等待并重试，保证程序健壮性。

#新的推流方式
ffmpeg -rtsp_transport tcp -i "rtsp://admin:123456@192.168.30.98:554/profile1" \
-vf scale=640:360 -f rawvideo -pix_fmt bgr24 udp://127.0.0.1:12345

#-------------------------------------------------------------------------------------------