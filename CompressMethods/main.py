# from image_processor_cuda import ImageProcessor

# def main():
#     # 创建 ImageProcessor 对象
#     processor = ImageProcessor(image_path="test7.bmp")
#     # 执行处理
#     decoded_x, decoded_y, decoded_z, restored_image = processor.process_image()

from image_processor_cuda_up import ImageProcessor
import tkinter as tk
from tkinter import filedialog

def main():
    """
    1、 RLE 压缩
    2、 DEFLATE 压缩
    3、 ZST压缩
    """

    # 初始化 tkinter 界面
    root = tk.Tk()
    root.withdraw()

    # 选择文件路径
    image_path = filedialog.askopenfilename(title="选择图像文件", filetypes=(("BMP Files", "*.bmp"), ("All Files", "*.*")))
    print(f"选择的文件路径为: {image_path}")

    if not image_path:
        print("未选择文件，程序退出！")
        return
    
    # 创建 ImageProcessor 对象
    processor = ImageProcessor(image_path = image_path)
    # 压缩方式映射
    compress_options = {
        "1": "rle",
        "2": "deflate",
        "3": "zst"
    }
    
    # 键盘输入压缩方式参数，直到输入有效
    while True:
        compress_method = input("请输入压缩方式（1: RLE, 2: DEFLATE, 3: ZST）：")
        
        if compress_method in compress_options:
            compress_method = compress_options[compress_method]
            print(f"选择的压缩方式是: {compress_method}")
            break  # 输入有效时跳出循环
        else:
            print("输入无效，请输入正确的压缩方式！")

    # 执行处理
    decoded_x_block, decoded_y_block, decoded_z_block, decoded_x_local, decoded_y_local, decoded_z_local, restored_image = processor.process_image(compress_method)

if __name__ == '__main__':
    main()



