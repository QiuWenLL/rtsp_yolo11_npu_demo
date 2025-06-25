import cv2
import argparse

# 使用 argparse 获取命令行参数
def parse_args():
    parser = argparse.ArgumentParser(description='RTSP Stream Viewer')
    parser.add_argument('rtsp_url', type=str, help='RTSP stream URL')
    return parser.parse_args()

def main():
    # 获取命令行参数
    args = parse_args()
    rtsp_url = args.rtsp_url  # 从命令行获取 RTSP 流 URL

    # 打开视频流
    cap = cv2.VideoCapture(rtsp_url)
    
    if not cap.isOpened():
        print(f"无法打开 RTSP 流：{rtsp_url}，请检查 FFmpeg 是否安装正确。")
        exit()

    while True:
        ret, frame = cap.read()
        if not ret:
            print("无法接收视频帧，可能是流中断。")
            break

        # 显示每一帧
        cv2.imshow('RTSP Stream', frame)

        # 如果按下 'q' 键则退出
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
