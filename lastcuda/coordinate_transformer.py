import numpy as np  
import time  
import math  
import pycuda.driver as cuda  
from pycuda.compiler import SourceModule  
import pycuda.autoinit 

class CoordinateTransformer:
    """
    坐标转换类，负责将图像的RGB值转换为三维坐标，或将三维坐标恢复为RGB值。
    """
    def __init__(self, N, width, height):
        """
        初始化CoordinateTransformer对象
        
        :param N: 三维坐标的分辨率，决定了坐标轴的大小
        :param width: 图像的宽度
        :param height: 图像的高度
        """
        self.N = N  # 三维坐标的分辨率（决定坐标值的范围）
        self.width = width  # 图像的宽度
        self.height = height  # 图像的高度

    def rgb_to_coordinates(self, image):
        """
        使用CUDA将RGB值转换为三维坐标。
        
        :param image: 输入的RGB图像数据，形状为 (height, width, 3)
        :return: 转换后的三维坐标 x_relative, y_relative, z_relative
        """
        start_time = time.time()  # 记录处理开始的时间

        # 定义CUDA内核代码，用于将RGB转换为坐标
        kernel_code = """
        __global__ void rgb_to_coordinates_cuda(unsigned char *image, float *x_relative, float *y_relative, float *z_relative, int N, int width, int height) {
            int x = blockIdx.x * blockDim.x + threadIdx.x;  // 计算当前线程对应的x坐标
            int y = blockIdx.y * blockDim.y + threadIdx.y;  // 计算当前线程对应的y坐标
            
            if (x < width && y < height) {
                int idx = (y * width + x) * 3;  // 计算图像中当前像素的RGB数据索引
                
                unsigned char R = image[idx];  // 获取红色分量
                unsigned char G = image[idx + 1];  // 获取绿色分量
                unsigned char B = image[idx + 2];  // 获取蓝色分量
                
                // 将RGB值映射到 [0, N-1] 范围内的坐标系
                // float x_val = roundf((float)R / 255.0f * (N - 1));  // 红色分量映射到 [0, N-1] 的块中点
                // float y_val = roundf((float)G / 255.0f * (N - 1));  // 绿色分量映射到 [0, N-1] 的块中点
                // float z_val = roundf((float)B / 255.0f * (N - 1));  // 蓝色分量映射到 [0, N-1] 的块中点
                float x_val = fminf(fmaxf((float)R / 255.0f * (float)(N - 1), 0.0f), (float)(N - 1));
                float y_val = fminf(fmaxf((float)G / 255.0f * (float)(N - 1), 0.0f), (float)(N - 1));
                float z_val = fminf(fmaxf((float)B / 255.0f * (float)(N - 1), 0.0f), (float)(N - 1));

                
                // 将映射后的坐标存储到相应的数组中
                x_relative[y * width + x] = x_val;
                y_relative[y * width + x] = y_val;
                z_relative[y * width + x] = z_val;
            }
        }
        """
        
        try:
            # 编译CUDA代码
            mod = SourceModule(kernel_code)  # 将CUDA代码编译为模块
            rgb_to_coordinates_kernel = mod.get_function("rgb_to_coordinates_cuda")  # 获取内核函数
        except Exception as e:
            print(f"CUDA kernel compilation failed: {str(e)}")
            raise  # 如果编译失败，抛出异常

        # 将图像数据从主机传输到GPU
        image_device = cuda.to_device(image)  # 使用pycuda的to_device方法将图像传到GPU内存

        # 初始化用于存储坐标的数组，并将它们分配到GPU
        x_relative_device = cuda.mem_alloc(self.width * self.height * 4)  # 为x坐标分配内存
        y_relative_device = cuda.mem_alloc(self.width * self.height * 4)  # 为y坐标分配内存
        z_relative_device = cuda.mem_alloc(self.width * self.height * 4)  # 为z坐标分配内存

        # 设置线程块和网格大小
        block_size = (16, 16, 1)  # 每个线程块的大小是16x16
        grid_size = (math.ceil(self.width / block_size[0]), 
                     math.ceil(self.height / block_size[1]))  # 根据图像尺寸确定网格大小

        # 启动CUDA内核
        rgb_to_coordinates_kernel(image_device, x_relative_device, y_relative_device, z_relative_device, 
                                  np.int32(self.N), np.int32(self.width), np.int32(self.height), 
                                  block=block_size, grid=grid_size)  # 启动内核进行并行计算

        # 从GPU内存复制结果到主机内存
        x_relative = np.zeros((self.height, self.width), dtype=np.float32)  # 创建用于存储结果的NumPy数组
        y_relative = np.zeros((self.height, self.width), dtype=np.float32)  # 创建用于存储结果的NumPy数组
        z_relative = np.zeros((self.height, self.width), dtype=np.float32)  # 创建用于存储结果的NumPy数组

        # 将GPU中的坐标数据复制回主机内存
        cuda.memcpy_dtoh(x_relative, x_relative_device)  
        cuda.memcpy_dtoh(y_relative, y_relative_device)  
        cuda.memcpy_dtoh(z_relative, z_relative_device)  

        end_time = time.time()  # 记录处理结束的时间
        print(f"RGB to coordinates conversion completed in {end_time - start_time:.2f} seconds.")  # 输出处理时间

        return x_relative, y_relative, z_relative

    def coordinates_to_rgb(self, x_relative, y_relative, z_relative):
        """
        使用CUDA将三维坐标恢复为RGB值。
        
        :param x_relative: x坐标
        :param y_relative: y坐标
        :param z_relative: z坐标
        :return: 恢复后的RGB图像
        """
        start_time = time.time()  # 记录处理开始的时间

        # 定义CUDA内核代码，用于将坐标恢复为RGB值
        kernel_code = """
        __global__ void coordinates_to_rgb_cuda(float *x_relative, float *y_relative, float *z_relative, int N, int width, int height, unsigned char *restored_image) {
            int x = blockIdx.x * blockDim.x + threadIdx.x;  // 计算当前线程对应的x坐标
            int y = blockIdx.y * blockDim.y + threadIdx.y;  // 计算当前线程对应的y坐标
            
            if (x < width && y < height) {
                int idx = (y * width + x) * 3;  // 计算图像中当前像素的RGB数据索引
                
                // 从坐标恢复RGB值
                unsigned char R = (unsigned char)(x_relative[y * width + x] * 255.0f / (N - 1));  // 恢复红色分量
                unsigned char G = (unsigned char)(y_relative[y * width + x] * 255.0f / (N - 1));  // 恢复绿色分量
                unsigned char B = (unsigned char)(z_relative[y * width + x] * 255.0f / (N - 1));  // 恢复蓝色分量
                
                // 将恢复后的RGB值存储到图像数组中
                restored_image[idx] = R;
                restored_image[idx + 1] = G;
                restored_image[idx + 2] = B;
            }
        }
        """
        
        try:
            # 编译CUDA代码
            mod = SourceModule(kernel_code)  # 将CUDA代码编译为模块
            coordinates_to_rgb_kernel = mod.get_function("coordinates_to_rgb_cuda")  # 获取内核函数
        except Exception as e:
            print(f"CUDA kernel compilation failed: {str(e)}")
            raise  # 如果编译失败，抛出异常
        
        # 将坐标数据传递到GPU
        x_relative_device = cuda.to_device(x_relative)
        y_relative_device = cuda.to_device(y_relative)
        z_relative_device = cuda.to_device(z_relative)
        
        # 初始化恢复后的图像，并将其分配到GPU
        restored_image_device = cuda.mem_alloc(self.width * self.height * 3)  # 为恢复后的图像分配内存
        
        # 设置线程块和网格大小
        block_size = (16, 16, 1)  # 每个线程块的大小
        grid_size = (math.ceil(self.width / block_size[0]), 
                     math.ceil(self.height / block_size[1]))  # 根据图像尺寸确定网格大小

        # 启动CUDA内核
        coordinates_to_rgb_kernel(x_relative_device, y_relative_device, z_relative_device, np.int32(self.N), 
                                  np.int32(self.width), np.int32(self.height), restored_image_device, 
                                  block=block_size, grid=grid_size)  # 启动内核进行并行计算
        
        # 从GPU内存获取恢复后的图像数据
        restored_image = np.zeros((self.height, self.width, 3), dtype=np.uint8)  # 创建空的图像数组
        cuda.memcpy_dtoh(restored_image, restored_image_device)  # 从GPU复制恢复后的图像到主机内存

        end_time = time.time()  # 记录处理结束的时间
        print(f"Coordinates to RGB conversion completed in {end_time - start_time:.2f} seconds.")  # 输出处理时间

        return restored_image
