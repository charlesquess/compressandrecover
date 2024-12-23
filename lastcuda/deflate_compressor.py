import numpy as np
import time
import zlib
import logging

class DEFcompressor:
    def __init__(self):
        logging.basicConfig(level=logging.INFO)

    def deflate_compress(self, data):
        """
        使用DEFLATE算法进行压缩
        :param data: 待压缩的数据
        :return: 压缩后的数据
        """
        # 记录开始时间
        start_time = time.time()
        
        # 数据转换为字节数组
        data_bytes = data.tobytes()
        
        # 对数据进行DEFLATE压缩
        compressed_data = zlib.compress(data_bytes)

        # 记录结束时间
        end_time = time.time()

        # 打印压缩时间
        logging.info("Compress time: {:.4f} s".format(end_time - start_time))

        return compressed_data

    def deflate_decompress(self, compressed_data):
        """
        使用DEFLATE算法进行解压缩
        :param compressed_data: 待解压缩的数据
        :return: 解压缩后的数据
        """
        # 记录开始时间
        start_time = time.time()

        # 对数据进行DEFLATE解压缩
        decompressed_data = zlib.decompress(compressed_data)

        # 数据转换为numpy数组
        decompressed_array = np.frombuffer(decompressed_data, dtype=np.uint8)

        # 记录结束时间
        end_time = time.time()

        # 打印解压缩时间
        logging.info("Decompress time: {:.4f} s".format(end_time - start_time))

        return decompressed_array
