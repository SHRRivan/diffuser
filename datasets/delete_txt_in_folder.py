#!/usr/bin/env python3
import os
import sys

def delete_txt_in_folder(folder: str, recursive: bool = False) -> None:
    if not os.path.isdir(folder):
        sys.exit(1)

    # 如果 recursive=True 就递归子目录
    if recursive:
        for root, _, files in os.walk(folder, topdown=False):
            for f in files:
                if f.lower().endswith('.txt'):
                    full = os.path.join(root, f)
                    try:
                        os.remove(full)
                        print('已删除 %s', full)
                    except OSError as e:
                        print.error('删除失败 %s : %s', full, e)
    else:
        for name in os.listdir(folder):
            if name.lower().endswith('.txt'):
                full = os.path.join(folder, name)
                if os.path.isfile(full):
                    try:
                        os.remove(full)
                        print.info('已删除 %s', full)
                    except OSError as e:
                        print.error('删除失败 %s : %s', full, e)


if __name__ == '__main__':
    if len(sys.argv) < 2:
        print('用法: python clean_txt.py 文件夹路径 [-r]')
        sys.exit(1)

    path = sys.argv[1]
    recurse = '-r' in sys.argv[2:]
    delete_txt_in_folder(path, recursive=recurse)