# from image_processor_cuda import ImageProcessor

# def main():
#     # 创建 ImageProcessor 对象
#     processor = ImageProcessor(image_path="test7.bmp")
#     # 执行处理
#     decoded_x, decoded_y, decoded_z, restored_image = processor.process_image()

from image_processor_cuda_up import ImageProcessor

def main():
    # 创建 ImageProcessor 对象
    processor = ImageProcessor(image_path="test7.bmp")
    # 键盘输入压缩方式参数
    """
    1、 RLE 压缩
    2、 DEFLATE 压缩
    3、 ZST压缩
    """
    compress_method = input("请输入压缩方式（1: RLE, 2: DEFLATE, 3: ZST）：")
    if compress_method == "1":
        compress_method = "rle"
    elif compress_method == "2":
        compress_method = "deflate"
    elif compress_method == "3":
        compress_method = "zst"
    else:
        print("请输入正确的压缩方式！")
        return

    # 执行处理
    decoded_x_block, decoded_y_block, decoded_z_block, decoded_x_local, decoded_y_local, decoded_z_local, restored_image = processor.process_image(compress_method)

if __name__ == '__main__':
    main()



