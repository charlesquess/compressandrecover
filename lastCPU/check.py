from PIL import Image
import numpy as np

def compare_images(image1_path, image2_path):
    # 加载两张图片
    image1 = Image.open(image1_path)
    image2 = Image.open(image2_path)

    # 将图片转换为RGB模式，以确保我们得到每个像素的RGB值
    image1 = image1.convert("RGB")
    image2 = image2.convert("RGB")

    # 将图片转换为NumPy数组，方便逐像素比较
    image1_array = np.array(image1)
    image2_array = np.array(image2)

    # 获取图像的尺寸
    height1, width1, _ = image1_array.shape
    height2, width2, _ = image2_array.shape

    # 确保图片的尺寸相同
    if height1 != height2 or width1 != width2:
        print("The two images have different sizes.")
        return

    # 遍历每个像素点并进行比较
    different_pixels = []
    for y in range(height1):
        for x in range(width1):
            # 比较两个图像相同位置的像素是否不同
            if not np.array_equal(image1_array[y, x], image2_array[y, x]):
                # 如果不同，记录该位置及其RGB值
                rgb_image1 = tuple(image1_array[y, x])
                rgb_image2 = tuple(image2_array[y, x])
                different_pixels.append(((x, y), rgb_image1, rgb_image2))

    # 打印不同像素的位置和对应的RGB值
    if different_pixels:
        print(f"Found {len(different_pixels)} different pixel(s).")
        # for (x, y), rgb1, rgb2 in different_pixels:
        #     print(f"Pixel at ({x}, {y}) -> Image1 RGB: {rgb1}, Image2 RGB: {rgb2}")
        # 总共有多少个不同的像素点
        print(f"Total different pixels: {len(different_pixels)}")
    else:
        print("No differences found between the two images.")

# 使用函数
image1_path = 'test3.bmp'  # 第一张图片路径
image2_path = 'restored_image.bmp'  # 第二张图片路径

compare_images(image1_path, image2_path)
