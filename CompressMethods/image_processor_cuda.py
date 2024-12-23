import numpy as np  
import logging  
import time  
from PIL import Image  
from pycuda.compiler import SourceModule  
import pycuda.driver as cuda  
import pycuda.autoinit  
import math  
import os  
import sys  
from rle_compressor import RLECompressor  
from deflate_compressor import DEFcompressor
from coordinate_transformer import CoordinateTransformer 

# 设置日志配置，记录调试信息
logging.basicConfig(level=logging.DEBUG, format="%(asctime)s - %(levelname)s - %(message)s")

class ImageProcessor:
    """
    图像处理类：负责图像的加载、坐标转换、压缩、解压以及恢复图像。
    """
    def __init__(self, image_path, N=16):
        """
        初始化ImageProcessor对象
        
        :param image_path: 输入图像的路径
        :param N: 三维坐标的分辨率
        """
        self.image_path = image_path  # 图像路径
        self.N = N  # 三维坐标的分辨率，用于坐标的映射
        self.image = None  # 图像数据，初始化为空
        self.width = None  # 图像的宽度，初始化为空
        self.height = None  # 图像的高度，初始化为空
        self.rle_compressor = RLECompressor()  # 创建RLECompressor对象，用于RLE压缩和解压
        self.def_compressor = DEFcompressor()  # 创建DEFCompressor对象，用于DEF压缩和解压
        self.coordinate_transformer = None  # 坐标转换器对象，初始化为空

        # 加载图像并获取其尺寸
        self.load_image()

        # 初始化坐标转换器，传入坐标分辨率和图像尺寸
        self.coordinate_transformer = CoordinateTransformer(self.N, self.width, self.height)

    def load_image(self):
        """
        加载图像并获取图像的尺寸信息
        """
        image = Image.open(self.image_path).convert("RGB")  # 使用PIL库打开图像文件
        self.image = np.array(image)  # 将图像转换为NumPy数组（高度x宽度x通道）
        self.height, self.width, _ = self.image.shape  # 获取图像的高度、宽度和通道数
        logging.info(f"Image loaded. Shape: {self.image.shape}, dtype: {self.image.dtype}")  # 记录图像信息

    def process_image(self):
        """
        主处理函数：执行整个图像处理流程
        1. RGB转换为坐标
        2. 压缩坐标（RLE压缩）
        3. 解压缩坐标
        4. 坐标转换回RGB
        5. 保存图像
        """
        logging.info("Start processing the image.")  # 记录处理开始信息
        start_time = time.time()  # 记录处理开始的时间
        
        # 执行RGB到坐标的转换，返回相对坐标x、y、z
        x_relative, y_relative, z_relative = self.coordinate_transformer.rgb_to_coordinates(self.image)

        # 打印相对坐标的内存占用
        print(f"压缩前的坐标数据所占用的空间大小 : {sys.getsizeof(x_relative)+ sys.getsizeof(y_relative)+ sys.getsizeof(z_relative)} bytes")

        # 执行RLE压缩，对x、y、z坐标分别进行压缩
        compressed_x = self.rle_compressor.rle_encode(x_relative)
        compressed_y = self.rle_compressor.rle_encode(y_relative)
        compressed_z = self.rle_compressor.rle_encode(z_relative)

        # # 执行DEF压缩，对x、y、z坐标分别进行压缩
        # compressed_x = self.def_compressor.deflate_compress(x_relative)
        # compressed_y = self.def_compressor.deflate_compress(y_relative)
        # compressed_z = self.def_compressor.deflate_compress(z_relative)

        print(f"压缩后的坐标数据所占用的空间大小 : {sys.getsizeof(compressed_x)+ sys.getsizeof(compressed_y)+ sys.getsizeof(compressed_z)} bytes")

        # 执行RLE解压，将压缩后的坐标数据解压回原来的坐标形式
        decoded_x = self.rle_compressor.rle_decode(compressed_x, self.width, self.height)
        decoded_y = self.rle_compressor.rle_decode(compressed_y, self.width, self.height)
        decoded_z = self.rle_compressor.rle_decode(compressed_z, self.width, self.height)

        # # 执行DEF解压，将压缩后的坐标数据解压回原来的坐标形式
        # decoded_x = self.def_compressor.deflate_decompress(compressed_x)
        # decoded_y = self.def_compressor.deflate_decompress(compressed_y)
        # decoded_z = self.def_compressor.deflate_decompress(compressed_z)
        
        # # CUDA加速部分
        # # 执行RLE压缩，对x、y、z坐标分别进行压缩
        # compressed_x, compressed_y, compressed_z = (
        #     self.rle_compressor.rle_encode(x_relative, self.width, self.height),
        #     self.rle_compressor.rle_encode(y_relative, self.width, self.height),
        #     self.rle_compressor.rle_encode(z_relative, self.width, self.height)
        # )

        # # print("compressed_x:", compressed_x)
        # # print("compressed_y:", compressed_y)    
        # # print("compressed_z:", compressed_z)

        # # 拆分压缩后的坐标数据，分别保存为counts和values
        # compressed_x_counts, compressed_x_values = compressed_x
        # compressed_y_counts, compressed_y_values = compressed_y
        # compressed_z_counts, compressed_z_values = compressed_z

        # print("compressed_x_counts:", compressed_x_counts)
        # print("compressed_x_values:", compressed_x_values)
        # print("compressed_y_counts:", compressed_y_counts)
        # print("compressed_y_values:", compressed_y_values)    
        # print("compressed_z_counts:", compressed_z_counts)
        # print("compressed_z_values:", compressed_z_values)


        # # 执行RLE解压，将压缩后的坐标数据解压回原来的坐标形式
        # # 调用 rle_decode 函数时传递 counts, values, width 和 height
        # decoded_x = self.rle_compressor.rle_decode(compressed_x_counts, compressed_x_values, self.width, self.height)
        # decoded_y = self.rle_compressor.rle_decode(compressed_y_counts, compressed_y_values, self.width, self.height)
        # decoded_z = self.rle_compressor.rle_decode(compressed_z_counts, compressed_z_values, self.width, self.height)

        # 执行坐标到RGB的转换，恢复图像
        restored_image = self.coordinate_transformer.coordinates_to_rgb(decoded_x, decoded_y, decoded_z)

        # 将恢复后的图像保存为BMP格式
        restored_image = Image.fromarray(restored_image)  # 将NumPy数组转换为PIL图像对象
        restored_image.save("restored_image.bmp")  # 保存恢复的图像
        logging.info("Image saved.")  # 记录保存成功的信息

        end_time = time.time()  # 记录处理结束的时间
        logging.info(f"Processing completed in {end_time - start_time:.2f} seconds.")  # 记录总的处理时间

        return decoded_x, decoded_y, decoded_z, restored_image  # 返回解压后的坐标数据和恢复的图像
