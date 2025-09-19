/*
请帮我构建基于cuda的图像叠加程序
1. 读取输入图像 path1 和 path2
2. 将img1和img2叠加
3. 将叠加后的图像保存为 path3
4. 使用cuda实现

编译命令：
vscode注意将opencv path添加到.vscode/c_cpp_properties.json
nvcc -o demo demo.cu `pkg-config --cflags --libs opencv4`
*/

#include <iostream>
#include <opencv4/opencv2/opencv.hpp>

using namespace cv;


// CUDA kernel：逐像素叠加
__global__ void addImageKernel(unsigned char* img1, unsigned char* img2, unsigned char* output, int width, int height, int channels) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    int index = (y * width + x) * channels;

    if (x < width && y < height) {
        for (int c = 0; c < channels; c++) {
            output[index + c] = img1[index + c] + img2[index + c];
        }
    }
}

int main(int argc, char** argv) {
    if (argc < 4) {
        std::cout << "Usage: ./demo path1 path2 path3" << std::endl;
        return -1;
    }

    std::string path1 = argv[1];
    std::string path2 = argv[2];
    std::string path3 = argv[3];

    // 读取输入图像 保持相同大小和通道数
    cv::Mat img1 = cv::imread(path1, cv::IMREAD_COLOR);
    cv::Mat img2 = cv::imread(path2, cv::IMREAD_COLOR);

    // 检查图像是否为空
    if (img1.empty() || img2.empty()) {
        std::cout << "Error: Failed to read input images." << std::endl;
        return -1;
    }

    // 检查图像大小是否一致
    if (img1.size() != img2.size()) {
        std::cout << "Error: Input images must have the same size." << std::endl;
        return -1;
    }

    int width = img1.cols;
    int height = img1.rows;
    int channels = img1.channels();
    size_t size = width * height * channels * sizeof(unsigned char);

    cv::Mat output(height, width, img1.type());

    // 分配GPU内存
    unsigned char* d_img1, * d_img2, * d_output;
    cudaMalloc((void**) &d_img1, size);
    cudaMalloc((void**) &d_img2, size);
    cudaMalloc((void**) &d_output, size);

    // 拷贝数据到GPU
    cudaMemcpy(d_img1, img1.data, size, cudaMemcpyHostToDevice);
    cudaMemcpy(d_img2, img2.data, size, cudaMemcpyHostToDevice);

    // 定义线程块和网络
    dim3 block(16, 16);
    dim3 grid((width + block.x - 1) / block.x, (height + block.y - 1) / block.y);

    // 调用cuda核
    addImageKernel <<< grid, block >>> (d_img1, d_img2, d_output, width, height, channels);

    cudaDeviceSynchronize();

    // 从GPU拷贝结果到主机
    cudaMemcpy(output.data, d_output, size, cudaMemcpyDeviceToHost);

    // 保存结果
    cv::imwrite(path3, output);

    // 释放GPU内存
    cudaFree(d_img1);
    cudaFree(d_img2);
    cudaFree(d_output);

    std::cout << "Image addition completed. Result saved to " << path3 << std::endl;

    return 0;
}