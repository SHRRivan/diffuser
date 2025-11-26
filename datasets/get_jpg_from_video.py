import cv2
import os
import shutil


if __name__ == '__main__':
    # 视频文件路径
    video_path = input("输入待处理的视频(包括后缀): ")
    # video_path = r'G:\华南理工大学\燃气管道智慧监测\视频\遮挡.mp4'

    # 输出文件夹路径
    output_dir = input("请输入你想要创建的文件夹名称(不要重名): ")

    # 初始化视频捕获对象
    cap = cv2.VideoCapture(video_path)

    # 检查是否成功打开视频文件
    if not cap.isOpened():
        print("Error: Unable to open the video.")
        exit()

    # 初始化总帧数帧计数器
    total_frames = cap.get(cv2.CAP_PROP_FRAME_COUNT)
    frame_count = 0
    frame_name = 1

    # 每隔10帧保存一次
    save_interval = 10

    if not cv2.os.path.exists(output_dir):
        cv2.os.makedirs(output_dir)
    else:
        try:
            os.removedirs(output_dir)  # 删除空文件夹
        except OSError:
            shutil.rmtree(output_dir)  # 递归删除文件夹，即：删除非空文件夹
            os.makedirs(output_dir)    # 重新建立文件夹

    # 读取并保存帧，直到视频结束
    while cap.isOpened():
        ret, frame = cap.read()

        # 如果成功读取帧
        if ret:
            # 检查是否到达保存间隔
            if frame_count % save_interval == 0:
                # 构造输出文件名
                output_file = cv2.os.path.join(output_dir, f'{video_path[:-4]}_{frame_name}.jpg')

                # 保存当前帧到文件
                cv2.imwrite(output_file, frame)
                print(f"Saved frame {frame_name} to {output_file}")

                frame_name += 1

            # 帧前进
            if total_frames - frame_count == 1:
                output_file = cv2.os.path.join(output_dir, f'{frame_name}.jpg')
                cv2.imwrite(output_file, frame)
                print(f"Saved frame {frame_name} to {output_file}")
            frame_count += 1
        else:
            # 如果没有更多帧，退出循环
            break

            # 释放视频捕获对象并关闭所有OpenCV窗口
    cap.release()
    cv2.destroyAllWindows()

    # print("请按任意键退出~")
    # ord(msvcrt.getch())
