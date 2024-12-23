import numpy as np
import logging
import time

# 设置日志
logging.basicConfig(level=logging.DEBUG, format="%(asctime)s - %(levelname)s - %(message)s")

class RLECompressor:
    """
    使用RLE（Run-Length Encoding）算法对图像进行压缩和解压缩的类。
    """
    def __init__(self):
        """
        初始化RLECompressor类。
        """
        pass

    def rle_encode(self, image):
        """
        使用RLE编码对图像进行压缩。

        :param image: 输入的图像数据，假设为一个二维NumPy数组，每个元素为一个像素（如[R, G, B]）。
        :return: 压缩后的数据，其中每一行包含若干 (count, pixel) 元组。
        """
        start_time = time.time()  # 记录编码开始时间
        compressed_data = []  # 用于存储压缩后的数据
        
        # 对图像的每一行进行RLE编码
        for row in image:
            # 对每一行进行编码
            encoded_row = []  # 当前行的编码结果
            pre_pixel = row[0]  # 记录当前行的第一个像素
            count = 1  # 初始像素计数为1

            # 遍历该行的剩余像素
            for pixel in row[1:]:
                if np.array_equal(pixel, pre_pixel):  # 如果当前像素与前一个像素相同
                    count += 1  # 增加计数
                else:
                    # 如果当前像素与前一个像素不同，保存当前像素的计数和像素值
                    encoded_row.append((count, pre_pixel))  
                    pre_pixel = pixel  # 更新当前像素为新的像素
                    count = 1  # 重置计数为1

            # 处理当前行的最后一个像素
            encoded_row.append((count, pre_pixel))  # 存储最后一个像素的计数和像素值
            
            # 将编码后的行数据添加到压缩数据列表
            compressed_data.append(encoded_row)
        
        end_time = time.time()  # 记录编码结束时间
        logging.info(f"RLE encoding completed in {end_time - start_time:.2f} seconds.")  # 记录日志
        
        return compressed_data

    def rle_decode(self, compressed_data):
        """
        使用RLE解码对图像进行解码。

        :param compressed_data: 输入的压缩数据，每个元素是一个包含若干 (count, pixel) 元组的列表。
        :return: 解码后的图像数据，格式为二维NumPy数组。
        """
        start_time = time.time()  # 记录解码开始时间
        decoded_image = []  # 初始化解码后的图像为空列表
        
        # 对压缩数据的每一行进行解码
        for encoded_row in compressed_data:
            row = []  # 当前行的解码结果
            for count, pixel in encoded_row:
                row.extend([pixel] * count)  # 根据计数将像素值重复添加到当前行
            decoded_image.append(np.array(row))  # 将解码后的行添加到解码图像中
        
        end_time = time.time()  # 记录解码结束时间
        logging.info(f"RLE decoding completed in {end_time - start_time:.2f} seconds.")  # 记录日志
        
        return np.array(decoded_image)
