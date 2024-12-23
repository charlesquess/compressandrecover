import numpy as np
import pycuda.driver as cuda
from pycuda.compiler import SourceModule
import pycuda.autoinit
import math
import time

class RLECompressor:
    """
    使用CUDA加速的RLE（Run-Length Encoding）压缩和解压类。
    """
    def __init__(self):
        pass

    def rle_encode(self, data, width, height):
        """
        使用CUDA加速的RLE编码压缩
        :param data: 输入数据，NumPy数组（高度x宽度）
        :param width: 图像宽度
        :param height: 图像高度
        :return: 压缩后的数据
        """
        kernel_code = """
        __global__ void rle_encode_cuda(unsigned char *data, int *counts, unsigned char *values, int width, int height) {
            int x = blockIdx.x * blockDim.x + threadIdx.x;
            int y = blockIdx.y * blockDim.y + threadIdx.y;

            if (x < width && y < height) {
                int idx = y * width + x;
                unsigned char current_value = data[idx];

                // 线程负责一个像素的值和计数
                int count = 1;  // 假设每个像素出现一次，后续可以优化以处理重复像素
                
                // 计算存储索引
                int index = y * width + x;  // 每个像素对应一个位置

                // 存储值和计数
                counts[index] = count;
                values[index] = current_value;
            }
        }
        """
        # 编译CUDA代码
        mod = SourceModule(kernel_code)
        rle_encode_kernel = mod.get_function("rle_encode_cuda")

        # 将输入数据传输到GPU
        data_device = cuda.to_device(data)

        # 分配内存用于存储压缩数据
        counts_device = cuda.mem_alloc(width * height * 4)  # 每个像素1个int
        values_device = cuda.mem_alloc(width * height)      # 每个像素1个unsigned char

        # 设置CUDA线程块和网格大小
        block_size = (16, 16, 1)
        grid_size = (math.ceil(width / block_size[0]), math.ceil(height / block_size[1]))

        # 启动内核进行压缩
        rle_encode_kernel(data_device, counts_device, values_device, np.int32(width), np.int32(height), block=block_size, grid=grid_size)

        # 从GPU内存获取结果
        counts = np.zeros((height, width), dtype=np.int32)
        values = np.zeros((height, width), dtype=np.uint8)

        cuda.memcpy_dtoh(counts, counts_device)
        cuda.memcpy_dtoh(values, values_device)

        return counts, values

    def rle_decode(self, counts, values, width, height):
        """
        使用CUDA加速的RLE解压缩
        :param counts: 压缩后的计数数据
        :param values: 压缩后的像素值数据
        :param width: 图像宽度
        :param height: 图像高度
        :return: 解压缩后的数据
        """
        kernel_code = """
        __global__ void rle_decode_cuda(int *counts, unsigned char *values, unsigned char *output, int width, int height) {
            int x = blockIdx.x * blockDim.x + threadIdx.x;
            int y = blockIdx.y * blockDim.y + threadIdx.y;

            if (x < width && y < height) {
                int idx = y * width + x;
                int count = counts[idx];  // 获取当前像素的重复次数
                unsigned char value = values[idx];  // 获取当前像素的值

                // 解压缩时，将像素值复制"count"次
                for (int i = 0; i < count; i++) {
                    if (idx + i < width * height) {
                        output[idx + i] = value;  // 将像素值写入输出图像
                    }
                }
            }
        }
        """
        # 编译CUDA代码
        mod = SourceModule(kernel_code)
        rle_decode_kernel = mod.get_function("rle_decode_cuda")

        # 将 counts 和 values 数据传到GPU上
        counts_device = cuda.to_device(counts)  # 将 counts 复制到 GPU 上
        values_device = cuda.to_device(values)  # 将 values 复制到 GPU 上

        # 分配内存用于解压数据
        output_device = cuda.mem_alloc(width * height) 

        # 设置CUDA线程块和网格大小
        block_size = (16, 16, 1)
        grid_size = (math.ceil(width / block_size[0]), math.ceil(height / block_size[1]))

        # 启动CUDA内核进行解压
        rle_decode_kernel(counts, values, output_device, np.int32(width), np.int32(height), block=block_size, grid=grid_size)

        # 从GPU内存获取解压后的结果
        output = np.zeros((height, width), dtype=np.uint8)
        cuda.memcpy_dtoh(output, output_device)

        return output
