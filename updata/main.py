from image_processor_cuda import ImageProcessor

def main():
    # 创建 ImageProcessor 对象
    processor = ImageProcessor(image_path="C:/Users/29814/CompressAndRecoverImage/updata/test.bmp")
    # 执行处理
    decoded_x, decoded_y, decoded_z, restored_image = processor.process_image()

if __name__ == '__main__':
    main()



