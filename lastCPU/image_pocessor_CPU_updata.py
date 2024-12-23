import numpy as np
import logging
import time
from PIL import Image

# 设置日志
logging.basicConfig(level=logging.DEBUG, format="%(asctime)s - %(levelname)s - %(message)s")

class ImageProcessor:
    def __init__(self, image_path, N=16):
        self.image_path = image_path
        self.N = N   # 块的数量
        self.image = None
        self.width = None
        self.height = None
        self.original_image = None  # 保存原始图像
        
        # 加载图像
        self.load_image()

    def load_image(self):
        """加载图像并获取其尺寸"""
        image = Image.open(self.image_path)
        self.image = np.array(image)  # 将图像转换为numpy数组
        self.original_image = np.copy(self.image)  # 保存原始图像
        self.height, self.width, _ = self.image.shape
        logging.info(f"Image loaded. Shape: {self.image.shape}, dtype: {self.image.dtype}")
    
    def rgb_to_coordinates(self):
        """将RGB值转换为坐标并记录日志"""
        # 初始化坐标数组
        block_x = np.zeros((self.height, self.width), dtype=np.uint8)
        block_y = np.zeros((self.height, self.width), dtype=np.uint8)
        block_z = np.zeros((self.height, self.width), dtype=np.uint8)

        local_x = np.zeros((self.height, self.width), dtype=np.uint8)
        local_y = np.zeros((self.height, self.width), dtype=np.uint8)
        local_z = np.zeros((self.height, self.width), dtype=np.uint8)
        

        # 逐像素处理
        for y in range(self.height):
            for x in range(self.width):
                # 获取当前像素的RGB值
                R, G, B = self.image[y, x]
                
                # 映射到块坐标系和局部坐标
                block_x_val = R // (256 // self.N) # x 轴的块坐标
                local_x_val = R % (256 // self.N)  # x 轴的局部坐标

                block_y_val = G // (256 // self.N) # y 轴的块坐标
                local_y_val = G % (256 // self.N)  # y 轴的局部坐标
 
                block_z_val = B // (256 // self.N) # z 轴的块坐标
                local_z_val = B % (256 // self.N)  # z 轴的局部坐标
                # 打印前几个像素的RGB和坐标
                if x < 5 and y == 0 :  # 仅打印前5个像素的数据，避免输出过多
                    logging.debug(f"Pixel({x}, {y}) RGB: ({R}, {G}, {B}) => "f"block_x: {block_x_val}, local_x: {local_x_val}, "f"block_y: {block_y_val}, local_y: {local_y_val}, "f"block_z: {block_z_val}, local_z: {local_z_val}")
                
                # 保存坐标
                block_x[y, x] = block_x_val
                block_y[y, x] = block_y_val
                block_z[y, x] = block_z_val
                local_x[y, x] = local_x_val
                local_y[y, x] = local_y_val
                local_z[y, x] = local_z_val

        # 返回每个通道的块坐标和局部坐标
        return block_x, block_y, block_z, local_x, local_y, local_z


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

    def rle_decode(self, compressed_data):
        """ 使用RLE解码对图像进行解压 """
        decoded_image = []  # 初始化解码图像为空列表
        
        for encoded_row in compressed_data:
            # 对每一行进行解码
            row = []
            for count, pixel in encoded_row:
                row.extend([pixel] * count)  # 根据计数解压像素值
            decoded_image.append(np.array(row))  # 将解码后的行添加到 decoded_image
            
        return np.array(decoded_image)

    def coordinates_to_rgb(self, block_x, block_y, block_z, local_x, local_y, local_z):
        """将坐标转换回RGB值"""
        restored_image = np.zeros((self.height, self.width, 3), dtype=np.uint8)

        # 逐像素处理
        for y in range(self.height):
            for x in range(self.width):
                # 映射回RGB值
                R = np.clip(round(block_x[y, x] * (256 // self.N) + local_x[y, x]), 0, 255)
                G = np.clip(round(block_y[y, x] * (256 // self.N) + local_y[y, x]), 0, 255)
                B = np.clip(round(block_z[y, x] * (256 // self.N) + local_z[y, x]), 0, 255)

                # 保存映射后的RGB值
                restored_image[y, x] = (R, G, B)

        return restored_image

    def compare_images(self, restored_image):
        """对比原始图像和恢复后的图像"""
        differences = np.sum(self.original_image != restored_image)
        total_pixels = self.height * self.width
        logging.info(f"Total pixels: {total_pixels}, Differences: {differences}")
        
        if differences > 0:
            logging.warning(f"There are differences between the original and restored images!")
        else:
            logging.info("The original and restored images are identical.")

    def compare_compressed_decoed(self, compressed_data, decoded_data):
        """对比压缩前和解压后的像素差异"""
        differences = np.sum(compressed_data != decoded_data)
        total_pixels = self.height * self.width
        logging.info(f"Total pixels: {total_pixels}, Differences: {differences}")
        
        if differences > 0:
            logging.warning(f"There are differences between the compressed and decompressed data!")
        else:
            logging.info("The compressed and decompressed data are identical.")


    def process_image(self):
        """主处理函数"""
        logging.info("Start processing the image.")
        start_time = time.time()
        
        # 执行RGB到块坐标系和局部坐标系的转换
        block_x, block_y, block_z, local_x, local_y, local_z = self.rgb_to_coordinates()

        # 执行RLE压缩
        # 块坐标压缩
        compressed_block_x = self.rle_encode(block_x)
        compressed_block_y = self.rle_encode(block_y)
        compressed_block_z = self.rle_encode(block_z)
        # 局部坐标压缩
        compressed_local_x = self.rle_encode(local_x)
        compressed_local_y = self.rle_encode(local_y)
        compressed_local_z = self.rle_encode(local_z)
        
        # 执行RLE解压
        # 块坐标解压
        decoded_block_x = self.rle_decode(compressed_block_x)
        decoded_block_y = self.rle_decode(compressed_block_y)
        decoded_block_z = self.rle_decode(compressed_block_z)
        # 局部坐标解压
        decoded_local_x = self.rle_decode(compressed_local_x)
        decoded_local_y = self.rle_decode(compressed_local_y)
        decoded_local_z = self.rle_decode(compressed_local_z)

        # 对比压缩前和解压后的像素差异
        if not np.array_equal(block_x, decoded_block_x) or not np.array_equal(block_y, decoded_block_y) or not np.array_equal(block_z, decoded_block_z) or not np.array_equal(local_x, decoded_local_x) or not np.array_equal(local_y, decoded_local_y) or not np.array_equal(local_z, decoded_local_z):
            logging.warning("The compressed and decompressed data are different.")
        else:
            logging.info("The compressed and decompressed data are identical.")

        # 恢复RGB图像
        restored_image = self.coordinates_to_rgb(decoded_block_x, decoded_block_y, decoded_block_z, decoded_local_x, decoded_local_y, decoded_local_z)

        # 对比原始图像和恢复后的图像
        self.compare_images(restored_image)

        # 保存处理后的图像
        restored_image = Image.fromarray(restored_image)
        restored_image.save("restored_image.bmp")
        logging.info("Image saved.")

        end_time = time.time()
        logging.info(f"Processing completed in {end_time - start_time:.2f} seconds.")

        return decoded_block_x, decoded_block_y, decoded_block_z, decoded_local_x, decoded_local_y, decoded_local_z, restored_image


if __name__ == "__main__":
    # 创建 ImageProcessor 对象
    processor = ImageProcessor(image_path="test8.bmp")
    
    # 执行处理
    decoded_block_x, decoded_block_y, decoded_block_z, decoded_local_x, decoded_local_y, decoded_local_z, restored_image = processor.process_image()

