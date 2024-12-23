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
from coordinate_transformer_updata import CoordinateTransformer
from rle_compressor import RLECompressor 
from deflate_compressor import DEFcompressor
from zst_compressor import ZstdCompressor

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
        self.deflate_compressor = DEFcompressor()  # 创建DeflateCompressor对象，用于Deflate压缩和解压
        self.zstd_compressor = ZstdCompressor()  # 创建ZstdCompressor对象，用于zstd压缩和解压

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

    def process_image(self, compress_method):
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
        x_block, y_block, z_block, x_local, y_local, z_local = self.coordinate_transformer.rgb_to_coordinates(self.image)

        # 计算压缩前的坐标数据所占用的空间大小
        print(f" 压缩前的坐标数据所占用的空间大小: {sys.getsizeof(x_block) + sys.getsizeof(y_block) + sys.getsizeof(z_block) + sys.getsizeof(x_local) + sys.getsizeof(y_local) + sys.getsizeof(z_local)} bytes")

        if compress_method == "rle":
            # 执行RLE压缩，对x、y、z坐标分别进行压缩
            compoessed_x_block = self.rle_compressor.rle_encode(x_block)
            compoessed_y_block = self.rle_compressor.rle_encode(y_block)
            compoessed_z_block = self.rle_compressor.rle_encode(z_block)
            compoessed_x_local = self.rle_compressor.rle_encode(x_local)
            compoessed_y_local = self.rle_compressor.rle_encode(y_local)
            compoessed_z_local = self.rle_compressor.rle_encode(z_local)

        elif compress_method == 'deflate':
            # 执行Deflate压缩，对x、y、z坐标分别进行压缩
            compoessed_x_block = self.deflate_compressor.deflate_compress(x_block)
            compoessed_y_block = self.deflate_compressor.deflate_compress(y_block)
            compoessed_z_block = self.deflate_compressor.deflate_compress(z_block)
            compoessed_x_local = self.deflate_compressor.deflate_compress(x_local)
            compoessed_y_local = self.deflate_compressor.deflate_compress(y_local)
            compoessed_z_local = self.deflate_compressor.deflate_compress(z_local)

        elif compress_method == 'zst':
            # 执行zstd压缩，对x、y、z坐标分别进行压缩
            compoessed_x_block = self.zstd_compressor.compress(x_block)
            compoessed_y_block = self.zstd_compressor.compress(y_block)
            compoessed_z_block = self.zstd_compressor.compress(z_block)
            compoessed_x_local = self.zstd_compressor.compress(x_local)
            compoessed_y_local = self.zstd_compressor.compress(y_local)
            compoessed_z_local = self.zstd_compressor.compress(z_local)

        # 计算总占用空间大小
        print(f"压缩后的坐标数据所占用的空间大小: {sys.getsizeof(compoessed_x_block) + sys.getsizeof(compoessed_y_block) + sys.getsizeof(compoessed_z_block) + sys.getsizeof(compoessed_x_local) + sys.getsizeof(compoessed_y_local) + sys.getsizeof(compoessed_z_local)} bytes")
        
        # 保存压缩后的坐标数据为
        if compress_method == 'rle':
            # 执行RLE解压，将压缩后的坐标数据解压回原来的坐标形式
            decoded_x_block = self.rle_compressor.rle_decode(compoessed_x_block, self.width, self.height)
            decoded_y_block = self.rle_compressor.rle_decode(compoessed_y_block, self.width, self.height)
            decoded_z_block = self.rle_compressor.rle_decode(compoessed_z_block, self.width, self.height)
            decoded_x_local = self.rle_compressor.rle_decode(compoessed_x_local, self.width, self.height)
            decoded_y_local = self.rle_compressor.rle_decode(compoessed_y_local, self.width, self.height)
            decoded_z_local = self.rle_compressor.rle_decode(compoessed_z_local, self.width, self.height)

        elif compress_method == 'deflate':
            # 执行Deflate解压，将压缩后的坐标数据解压回原来的坐标形式
            decoded_x_block = self.deflate_compressor.deflate_decompress(compoessed_x_block)
            decoded_y_block = self.deflate_compressor.deflate_decompress(compoessed_y_block)
            decoded_z_block = self.deflate_compressor.deflate_decompress(compoessed_z_block)
            decoded_x_local = self.deflate_compressor.deflate_decompress(compoessed_x_local)
            decoded_y_local = self.deflate_compressor.deflate_decompress(compoessed_y_local)
            decoded_z_local = self.deflate_compressor.deflate_decompress(compoessed_z_local)

        elif compress_method == 'zst':
            # 执行zstd解压，将压缩后的坐标数据解压回原来的坐标形式
            decoded_x_block = self.zstd_compressor.decompress(compoessed_x_block)
            decoded_y_block = self.zstd_compressor.decompress(compoessed_y_block)
            decoded_z_block = self.zstd_compressor.decompress(compoessed_z_block)
            decoded_x_local = self.zstd_compressor.decompress(compoessed_x_local)
            decoded_y_local = self.zstd_compressor.decompress(compoessed_y_local)
            decoded_z_local = self.zstd_compressor.decompress(compoessed_z_local)
        
        # 查询使用的压缩方法
        print(f"Used compress method: {compress_method}")

        # 执行坐标到RGB的转换，恢复图像
        restored_image = self.coordinate_transformer.coordinates_to_rgb(decoded_x_block, decoded_y_block, decoded_z_block, decoded_x_local, decoded_y_local, decoded_z_local)

        # 将恢复后的图像保存为BMP格式
        restored_image = Image.fromarray(restored_image)  # 将NumPy数组转换为PIL图像对象
        restored_image.save("restored_image.bmp")  # 保存恢复的图像
        logging.info("Image saved.")  # 记录保存成功的信息

        end_time = time.time()  # 记录处理结束的时间
        logging.info(f"Processing completed in {end_time - start_time:.2f} seconds.")  # 记录总的处理时间

        return decoded_x_block, decoded_y_block, decoded_z_block, decoded_x_local, decoded_y_local, decoded_z_local, restored_image  # 返回解压后的坐标数据和恢复的图像
