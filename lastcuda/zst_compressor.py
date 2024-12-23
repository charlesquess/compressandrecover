import zstandard as zstd
import numpy as np
import time
import logging

class ZstdCompressor:
    def __init__(self, level=3):
        logging.basicConfig(level=logging.INFO)
        self.zestd_compressor = zstd.ZstdCompressor(level=level)  # 创建zstd压缩器对象，支持设置压缩级别
        self.zestd_decompressor = zstd.ZstdDecompressor()  # 创建zstd解压器对象

    def compress(self, data):
        try:
            start_time = time.perf_counter()  # 使用更精确的计时方法
            compressed_data = self.zestd_compressor.compress(data)  # 压缩数据
            end_time = time.perf_counter()
            logging.info(f"压缩数据大小：{len(compressed_data)}，压缩用时：{end_time - start_time:.4f}秒")
            logging.info(f"Original data size: {len(data)}bytes, Compressed data size: {len(compressed_data)}bytes")
            return compressed_data
        except Exception as e:
            logging.error(f"Compression failed: {e}")
            raise

    def decompress(self, compressed_data):
        try:
            start_time = time.perf_counter()
            decompressed_data = self.zestd_decompressor.decompress(compressed_data)  # 解压数据
            end_time = time.perf_counter()
            logging.info(f"解压数据大小：{len(decompressed_data)}，解压用时：{end_time - start_time:.4f}秒")
            logging.info(f"Compressed data size: {len(compressed_data)}bytes, Decompressed data size: {len(decompressed_data)}bytes")
            return decompressed_data
        except Exception as e:
            logging.error(f"Decompression failed: {e}")
            raise

if __name__ == '__main__':
    compressor = ZstdCompressor(level=3)  # 创建ZstdCompressor对象，指定压缩级别
    data = np.random.bytes(1024*1024)  # 生成1MB的随机数据

    compressed_data = compressor.compress(data)  # 压缩数据
    decompressed_data = compressor.decompress(compressed_data)  # 解压数据

    # 校验数据是否一致
    assert data == decompressed_data, "Original and decompressed data do not match!"  

    logging.info("压缩和解压成功！")
