1 CompressMethods\coordinate_transformer_updata.py
    1.1rgb_to_coordinates函数
    此函数用于将RGB坐标转换为新坐标，坐标转换原则为：
    原坐标为(x,y), (r,g,b)
    新坐标为(x,y), (r_block, g_block, b_block),(r_local, g_local, b_local)
    其中r_block, g_block, b_block为块坐标，r_local, g_local, b_local为局部坐标
    具体实现见代码
    1.2coordinates_to_rgb函数
    此函数用于将新坐标转换为RGB坐标，坐标转换原则为：
    原坐标为(x,y), (r_block, g_block, b_block),(r_local, g_local, b_local)
    新坐标为(x,y), (r,g,b)
    具体实现见代码
2 CompressMethods\coor_transformer_update.py
    2.1rgb_to_coordinates函数
    此函数用于将RGB坐标转换为新坐标，坐标转换原则为：
    原坐标为(x,y), (r,g,b)
    新坐标为(x,y), （r_new, g_new, b_new）
    其中r_new, g_new, b_new为新的RGB值(浮点数)
    具体实现见代码
    2.2coordinates_to_rgb函数
    此函数用于将新坐标转换为RGB坐标，坐标转换原则为：
    原坐标为(x,y), （r_new, g_new, b_new）
    新坐标为(x,y), (r,g,b)
    具体实现见代码
3 CompressMethods\defate_compressor.py
    3.1deflate_compress函数
    此函数用于对图像进行deflate压缩，具体实现见代码
    3.2deflate_decompress函数
    此函数用于对图像进行deflate解压，具体实现见代码
    3.3deflate压缩原理
    https://blog.csdn.net/tuwenqi2013/article/details/103758292
4 CompressMethods\image_processor_cuda_up.py
    4.1load_image函数
    此函数用于加载图像，具体实现见代码
    4.2process_image函数
    主处理函数：执行整个图像处理流程
        1. RGB转换为坐标
        2. 压缩坐标（RLE压缩）
        3. 解压缩坐标
        4. 坐标转换回RGB
        5. 保存图像
5 CompressMethods\image_processor_cuda.py
    5.1load_image函数
    此函数用于加载图像，具体实现见代码
    5.2process_image函数
    主处理函数：执行整个图像处理流程
        1. RGB转换为坐标
        2. 压缩坐标（RLE压缩）
        3. 解压缩坐标
        4. 坐标转换回RGB
        5. 保存图像
6 CompressMethods\main.py
   程序入口
7 CompressMethods\rle_compressor.py
    8.1rle_compress函数
    此函数用于对坐标进行RLE压缩，具体实现见代码
    8.2rle_decompress函数
    此函数用于对坐标进行RLE解压，具体实现见代码
    8.3RLE压缩原理
    https://blog.csdn.net/qq_41577650/article/details/135171038?ops_request_misc=%257B%2522request%255Fid%2522%253A%25228a159599f3231cfc6cfe4f1a9605f96f%2522%252C%2522scm%2522%253A%252220140713.130102334..%2522%257D&request_id=8a159599f3231cfc6cfe4f1a9605f96f&biz_id=0&utm_medium=distribute.pc_search_result.none-task-blog-2~all~sobaiduend~default-2-135171038-null-null.142^v100^pc_search_result_base5&utm_term=RLE%E5%8E%8B%E7%BC%A9&spm=1018.2226.3001.4187
8 CompressMethods\zst_compressor.py
    9.1zst_compress函数
    此函数用于对图像进行zst压缩，具体实现见代码
    9.2zst_decompress函数
    此函数用于对图像进行zst解压，具体实现见代码
    9.3zst压缩原理
    https://blog.csdn.net/qq_35667076/article/details/136341995?utm_medium=distribute.pc_relevant.none-task-blog-2~default~baidujs_utm_term~default-0-136341995-blog-109028968.235^v43^control&spm=1001.2101.3001.4242.1&utm_relevant_index=2