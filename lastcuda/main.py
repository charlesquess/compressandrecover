from image_processor_cuda import ImageProcessor

def main():
    # 创建 ImageProcessor 对象
    processor = ImageProcessor(image_path="test7.bmp")
    # 执行处理
    decoded_x, decoded_y, decoded_z, restored_image = processor.process_image()

# from image_processor_cuda_up import ImageProcessor

# def main():
#     # 创建 ImageProcessor 对象
#     processor = ImageProcessor(image_path="test7.bmp")
#     # 执行处理
#     decoded_x_block, decoded_y_block, decoded_z_block, decoded_x_local, decoded_y_local, decoded_z_local, restored_image = processor.process_image()

if __name__ == '__main__':
    main()



