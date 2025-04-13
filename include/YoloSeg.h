#ifndef YOlOSEG_H
#define YOlOSEG_H
#include <opencv2/opencv.hpp>
#include "Thirdparty/ncnn/ncnn/net.h"
#include "Thirdparty/ncnn/ncnn/layer.h"
#include <vector>
#include <fstream>
#include <chrono>

struct SegResult {
    cv::Rect_<float> bbox;    // 检测框 (x, y, w, h)
    int class_id;              // 类别ID
    float confidence;          // 置信度
    cv::Mat mask;              // 二值掩膜
    std::string class_name;    // 类别名称
    std::vector<float> mask_coeffs; // 掩膜系数
};

namespace ORB_SLAM2 
{

class Tracking;

class YoloSeg {
public:
    YoloSeg(const std::string& param_path, 
            const std::string& bin_path,
            int input_size = 640,
            float conf_thresh = 0.6f,
            float mask_thresh = 0.6f);

    /**
     * @brief 执行推理
     * @param image 输入图像 (cv::Mat格式)
     * @details 该函数会执行模型推理，并将结果存储在成员变量中
     * @note 注意：该函数不会执行后处理，后处理需要单独调用postprocess()函数
     */
    void DetectAndSeg(const cv::Mat& image);
    /**
     * @brief 执行后处理
     * @details 该函数会解析检测框，执行NMS，生成掩膜，并恢复掩膜到原始图像大小
     * @note 注意：该函数需要在inference()之后调用
     */
    void postprocess();
    /**
     * @brief 实例分割线程
     * @details 该函数会在一个独立的线程中运行，持续检测新图像
     */
    void run();
    void SetTracker(Tracking* tracker);
    /**
     * @brief 获取检测结果
     * @return 检测结果向量
     * @details 该函数返回当前检测到的所有结果，包括检测框、置信度、类别ID等信息
     */
    const std::vector<SegResult>& get_detections();
    void DrawSegmentation(cv::Mat& image);

private:

    /**
     * @brief 预处理函数
     * @param image 输入图像 (cv::Mat格式)
     * @details 该函数会将输入图像进行预处理，包括缩放、归一化等操作
     */
    void preprocess(const cv::Mat& image);

    /**
     * @brief 解析检测框
     * @param num_classes 类别数量
     * @param mask_dim 掩膜维度
     * @details 该函数会解析模型输出的检测框，并将结果存储在成员变量中
     */
    void parse_detections(int num_classes, int mask_dim);

    /**
     * @brief 执行NMS
     * @param iou_threshold IoU阈值
     * @param conf_threshold 置信度阈值
     * @details 该函数会执行NMS，去除重叠的检测框，过滤低置信度的框和同类且距离过近的框
     * @note 注意：该函数会修改成员变量detections_，只保留经过NMS处理后的检测框
     */
    void apply_yolo_style_nms(float iou_threshold = 0.45f, 
                              float conf_threshold = 0.35f);
    /**
     * @brief 生成掩膜
     * @details 该函数会根据检测框和掩膜系数生成掩膜，并将结果存储在成员变量中
     */
    void generate_masks();

    /**
     * @brief 恢复掩膜到原始图像大小
     * @details 将掩膜从160x160恢复到原始图像大小（480x640），并裁剪掉Letterbox填充部分
     * @note 输入掩膜尺寸：160x160 → 输出掩膜尺寸：480x640（直接对齐原图）
     */
    void restore_masks();
    
    /**
     * @brief 计算两个矩形框的DIoU (Distance-IoU)
     * @param a 第一个矩形框 (OpenCV Rect格式)
     * @param b 第二个矩形框 (OpenCV Rect格式)
     * @return DIoU值，范围[-1, 1]（1表示完全重合，<=0表示无重叠）
     */
    float calculate_diou(const cv::Rect_<float>& a, const cv::Rect_<float>& b);
    float sigmoid(float x);

    bool isNewImageArrived();

    void ImageSegFinished();
private:
    ncnn::Net net_;
    int input_size_;
    float conf_thresh_;
    float mask_thresh_;
    int orig_width_;
    int orig_height_;
    ncnn::Mat input_mat_;
    ncnn::Mat output_;
    ncnn::Mat mask_proto_;
    std::vector<SegResult> detections_;
    std::vector<SegResult> detections_to_show_;
    std::mutex mMutex;

public:
    cv::Mat mImageToSeg;
    std::mutex mMutexGetNewImage;
    std::mutex mMutexImageSegFinished;
    bool mbNewImageFlag;
    bool mbPotentialDynamicRegionExist;
    Tracking* mpTracker;
    std::vector<SegResult> mvPotentialDynamicSegResults;
    // COCO类别标签
    const std::vector<std::string> coco_classes_ = {
        "person", "bicycle", "car", "motorcycle", "airplane", "bus", "train", "truck", "boat", "traffic light",
        "fire hydrant", "stop sign", "parking meter", "bench", "bird", "cat", "dog", "horse", "sheep", "cow",
        "elephant", "bear", "zebra", "giraffe", "backpack", "umbrella", "handbag", "tie", "suitcase", "frisbee",
        "skis", "snowboard", "sports ball", "kite", "baseball bat", "baseball glove", "skateboard", "surfboard",
        "tennis racket", "bottle", "wine glass", "cup", "fork", "knife", "spoon", "bowl", "banana", "apple",
        "sandwich", "orange", "broccoli", "carrot", "hot dog", "pizza", "donut", "cake", "chair", "couch",
        "potted plant", "bed", "dining table", "toilet", "tv", "laptop", "mouse", "remote", "keyboard", "cell phone",
        "microwave", "oven", "toaster", "sink", "refrigerator", "book", "clock", "vase", "scissors", "teddy bear",
        "hair drier", "toothbrush"
    };
    // const std::vector<int> mvPotentialDynamicId = {
    //     0, 1, 2, 3, 4, 5, 6, 7, 8, 14, 15, 16, 17, 41, 45, 56, 66
    // };
    const unsigned char colors_[80][3] = {
        {56,  0,   255},
        {226, 255, 0},
        {0,   94,  255},
        {0,   37,  255},
        {0,   255, 94},
        {255, 226, 0},
        {0,   18,  255},
        {255, 151, 0},
        {170, 0,   255},
        {0,   255, 56},
        {255, 0,   75},
        {0,   75,  255},
        {0,   255, 169},
        {255, 0,   207},
        {75,  255, 0},
        {207, 0,   255},
        {37,  0,   255},
        {0,   207, 255},
        {94,  0,   255},
        {0,   255, 113},
        {255, 18,  0},
        {255, 0,   56},
        {18,  0,   255},
        {0,   255, 226},
        {170, 255, 0},
        {255, 0,   245},
        {151, 255, 0},
        {132, 255, 0},
        {75,  0,   255},
        {151, 0,   255},
        {0,   151, 255},
        {132, 0,   255},
        {0,   255, 245},
        {255, 132, 0},
        {226, 0,   255},
        {255, 37,  0},
        {207, 255, 0},
        {0,   255, 207},
        {94,  255, 0},
        {0,   226, 255},
        {56,  255, 0},
        {255, 94,  0},
        {255, 113, 0},
        {0,   132, 255},
        {255, 0,   132},
        {255, 170, 0},
        {255, 0,   188},
        {113, 255, 0},
        {245, 0,   255},
        {113, 0,   255},
        {255, 188, 0},
        {0,   113, 255},
        {255, 0,   0},
        {0,   56,  255},
        {255, 0,   113},
        {0,   255, 188},
        {255, 0,   94},
        {255, 0,   18},
        {18,  255, 0},
        {0,   255, 132},
        {0,   188, 255},
        {0,   245, 255},
        {0,   169, 255},
        {37,  255, 0},
        {255, 0,   151},
        {188, 0,   255},
        {0,   255, 37},
        {0,   255, 0},
        {255, 0,   170},
        {255, 0,   37},
        {255, 75,  0},
        {0,   0,   255},
        {255, 207, 0},
        {255, 0,   226},
        {255, 245, 0},
        {188, 255, 0},
        {0,   255, 18},
        {0,   255, 75},
        {0,   255, 151},
        {255, 56,  0},
    };
};
} // namespace ORB_SLAM2
#endif // YOlOSEG_H