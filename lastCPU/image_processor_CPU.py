import numpy as np
import logging
import time
from PIL import Image

# 设置日志
logging.basicConfig(level=logging.DEBUG, format="%(asctime)s - %(levelname)s - %(message)s")

class ImageProcessor:
    def __init__(self, image_path, N=50):
        self.image_path = image_path
        self.N = N  # 三维坐标的分辨率
        self.image = None
        self.width = None
        self.height = None
        
        # 加载图像
        self.load_image()

    def load_image(self):
        """加载图像并获取其尺寸"""
        image = Image.open(self.image_path)
        self.image = np.array(image)  # 将图像转换为numpy数组
        self.height, self.width, _ = self.image.shape
        logging.info(f"Image loaded. Shape: {self.image.shape}, dtype: {self.image.dtype}")
    
    def rgb_to_coordinates(self):
        """将RGB值转换为坐标并记录日志"""
        # 初始化坐标数组
        x_relative = np.zeros((self.height, self.width), dtype=np.uint8)
        y_relative = np.zeros((self.height, self.width), dtype=np.uint8)
        z_relative = np.zeros((self.height, self.width), dtype=np.uint8)
    

        # 逐像素处理
        for y in range(self.height):
            for x in range(self.width):
                # 获取当前像素的RGB值
                R, G, B = self.image[y, x]
                
                # 映射到坐标系
                x_val = np.clip(R / 255.0 * (self.N - 1), 0, self.N - 1)
                y_val = np.clip(G / 255.0 * (self.N - 1), 0, self.N - 1)
                z_val = np.clip(B / 255.0 * (self.N - 1), 0, self.N - 1)

                # 保存映射后的坐标值
                x_relative[y, x] = x_val
                y_relative[y, x] = y_val
                z_relative[y, x] = z_val
                
                # 打印前几个像素的RGB和坐标
                if x < 495 and y < 5 and x >490 :  # 仅打印前5个像素的数据，避免输出过多
                    logging.debug(f"Pixel({x}, {y}) RGB: ({R}, {G}, {B}) => x: {x_val}, y: {y_val}, z: {z_val}")
        
        return x_relative, y_relative, z_relative

    def rle_encode(self, image):
        """ 使用RLE编码对图像进行压缩 """
        compressed_data = []
        
        for row in image:
            # 对每一行进行编码
            encoded_row = []
            pre_pixel = row[0]
            count = 1

            for pixel in row[1:]:
                if np.array_equal(pixel, pre_pixel):  # 这里修正为 np.array_equal
                    count += 1
                else:
                    encoded_row.append((count, pre_pixel))  # 存储计数和像素值
                    pre_pixel = pixel
                    count = 1
            
            # 最后一行的编码
            encoded_row.append((count, pre_pixel))  # 存储计数和像素值
            
            # 压缩后的行数据
            compressed_data.append(encoded_row)
        
        return compressed_data


    def re_decode(self, compressed_data):
        """ 使用RLE解码对图像进行解压 """
        decoded_image = []  # 初始化解码图像为空列表
        
        for encoded_row in compressed_data:
            # 对每一行进行解码
            row = []
            for count, pixel in encoded_row:
                row.extend([pixel] * count)  # 根据计数解压像素值
            decoded_image.append(np.array(row))  # 将解码后的行添加到 decoded_image
            
        return np.array(decoded_image)

    def coordinates_to_rgb(self, x_relative, y_relative, z_relative):
        """将坐标转换回RGB值"""
        restored_image = np.zeros((self.height, self.width, 3), dtype=np.uint8)

        # 逐像素处理
        for y in range(self.height):
            for x in range(self.width):
                # 映射回RGB值
                R = np.clip(x_relative[y, x] / (self.N - 1) * 255, 0, 255).astype(np.uint8)
                G = np.clip(y_relative[y, x] / (self.N - 1) * 255, 0, 255).astype(np.uint8)
                B = np.clip(z_relative[y, x] / (self.N - 1) * 255, 0, 255).astype(np.uint8)

                # 保存映射后的RGB值
                restored_image[y, x] = (R, G, B)

        return restored_image


    def process_image(self):
        """主处理函数"""
        logging.info("Start processing the image.")
        start_time = time.time()
        
        # 执行RGB到坐标的转换
        x_relative, y_relative, z_relative = self.rgb_to_coordinates()

        # 执行RLE压缩
        compressed_x = self.rle_encode(x_relative)
        compressed_y = self.rle_encode(y_relative)
        compressed_z = self.rle_encode(z_relative)

        # 

        #输出压缩后的坐标数据
        for i in range(5):
            logging.debug(f"Compressed x[{i}]: {compressed_x[i]}"f" Compressed y[{i}]: {compressed_y[i]}"f"Compressed z[{i}]: {compressed_z[i]}\n")

        # 执行RLE解压
        decoded_x = self.re_decode(compressed_x)
        decoded_y = self.re_decode(compressed_y)
        decoded_z = self.re_decode(compressed_z)

        # 输出解压后的坐标数据
        for i in range(5):
            logging.debug(f"Decoded x[{i}]: {decoded_x[i]}"f"Decoded y[{i}]: {decoded_y[i]}"f"Decoded z[{i}]: {decoded_z[i]}\n")

        restored_image = self.coordinates_to_rgb(decoded_x, decoded_y, decoded_z)

        # 保存处理后的图像
        restored_image = Image.fromarray(restored_image)
        restored_image.save("restored_image.bmp")
        logging.info("Image saved.")

        end_time = time.time()
        logging.info(f"Processing completed in {end_time - start_time:.2f} seconds.")

        return decoded_x, decoded_y, decoded_z, restored_image


if __name__ == "__main__":
    # 创建 ImageProcessor 对象
    processor = ImageProcessor(image_path="test3.bmp")
    
    # 执行处理
    decoded_x, decoded_y, decoded_z, restored_image = processor.process_image()

    # 可选：显示处理后的坐标数据，或者将其转换回图像并保存
    logging.info("Decoded coordinates (x, y, z) completed.")
