#include "YoloSeg.h"
#include "Tracking.h"

namespace ORB_SLAM2
{
    YoloSeg::YoloSeg(const std::string& param_path, 
                     const std::string& bin_path,
                     int input_size,
                     float conf_thresh,
                     float mask_thresh)
        : input_size_(input_size),
          conf_thresh_(conf_thresh),
          mask_thresh_(mask_thresh) {
        // 加载模型
        net_.opt.use_vulkan_compute = true;
        if (net_.load_param(param_path.c_str())) {
            throw std::runtime_error("Failed to load param file");
        }
        if (net_.load_model(bin_path.c_str())) {
            throw std::runtime_error("Failed to load bin file");
        }
        mbNewImageFlag = false;
    }
    const std::vector<SegResult>& YoloSeg::get_detections() {
        return detections_;
    }
    void YoloSeg::DetectAndSeg(const cv::Mat& image) {
        // 清空旧结果
        detections_.clear();
        
        // 预处理
        preprocess(image);
        
        // 创建Extractor
        ncnn::Extractor ex = net_.create_extractor();
        ex.set_light_mode(true);
        ex.input("in0", input_mat_);
    
        ex.extract("out0", output_);  // 检测框输出
        ex.extract("out1", mask_proto_); // 掩膜输出
    
        postprocess();
        {
            std::unique_lock<std::mutex> lock(mMutex);
            detections_.swap(detections_to_show_);
        }
    }
    
    void YoloSeg::preprocess(const cv::Mat& image) {
        orig_height_ = image.rows;
        orig_width_ = image.cols;
    
        cv::Mat img_pre(input_size_, input_size_, CV_8UC3, cv::Scalar(114, 114, 114));
        image.copyTo(img_pre(cv::Rect(0, 80, image.cols, image.rows)));
        // 转换为ncnn::Mat
        input_mat_ = ncnn::Mat::from_pixels(img_pre.data, ncnn::Mat::PIXEL_BGR2RGB, img_pre.cols, img_pre.rows);
        // 归一化
        const float norm_vals[3] = {1/255.f, 1/255.f, 1/255.f};
        input_mat_.substract_mean_normalize(nullptr, norm_vals);
    }
    
    void YoloSeg::postprocess() {
        const int num_classes = 80;
        const int mask_dim = 32;
        // 1. 解析检测框
        parse_detections(num_classes, mask_dim);
    
        // 2. 执行NMS
        apply_yolo_style_nms(0.5, conf_thresh_);
    
        // 3. 生成掩膜
        generate_masks();
    
        // 4. 恢复掩膜到原始图像大小
        restore_masks();

        mvPotentialDynamicSegResults.clear();
        // 5. 保存潜在动态区域
        for(auto it = detections_.begin(); it != detections_.end(); ++it)
        {
            if(it->class_id == 0 || it->class_id == 56 || it->class_id == 66)
            {
                if(it->class_id == 0)
                {
                    // 对人的掩膜进行膨胀
                    cv::Mat kernel = cv::getStructuringElement(cv::MORPH_ELLIPSE, cv::Size(5, 5));
                    cv::dilate(it->mask, it->mask, kernel);
                }
                mvPotentialDynamicSegResults.push_back(*it);
            }
        }
        mbPotentialDynamicRegionExist = !mvPotentialDynamicSegResults.empty();
    }
    
    void YoloSeg::parse_detections(int num_classes, int mask_dim) {
        // output_维度应为 (w=8400, h=116, c=1)
        const int w = output_.w;  // 8400
        const int h = output_.h;    // 116
    
        // 1. 转置输出数据
        cv::Mat a(h, w, CV_32FC1);
        memcpy((uchar*)a.data, output_.data, w * h * sizeof(float));
        cv::Mat mat = a.t();
        
        // 2. 解析检测框
        for(int i = 0; i < mat.rows; ++i) {
            // 前四维代表坐标
            const float* ptr = mat.ptr<float>(i);
            float cx = mat.at<float>(i, 0), cy = mat.at<float>(i, 1), width = mat.at<float>(i, 2), height = mat.at<float>(i, 3);
            // 如果bounding box的面积小于2500，即使该物体是动态的，也不进行处理
            if (width * height < 3000) continue;
            // 第4维开始是类别得分
            int class_id = std::max_element(ptr+4, ptr+4+num_classes) - (ptr+4);
            float class_score = mat.at<float>(i, 4 + class_id);
            float threshold = conf_thresh_;
            if(class_id == 0)
                threshold = 0.35;
            if (class_score < threshold) continue;
            float score_object = class_score;  // 使用sigmoid而不是softmax
    
            // 掩膜系数 (最后32维)
            std::vector<float> mask_coeffs(ptr + 4 + num_classes, ptr + 4 + num_classes + mask_dim);
    
            // 保存结果，注意letterbox坐标需要转换为(x, y, w, h)格式
            float x = cx - width / 2;
            float y = cy - height / 2 - 80;
            x = std::max(0.f, x);
            y = std::max(0.f, y);
            width = std::min(orig_width_ - x, width);
            height = std::min(orig_height_ - y, height);
            SegResult res;
            res.bbox = cv::Rect_<float>(x, y, width, height);
            res.class_id = class_id;
            res.confidence = score_object;
            res.mask_coeffs = std::move(mask_coeffs);
            res.class_name = coco_classes_[class_id];
            this->detections_.push_back(std::move(res));
        }
    }
    
    
    void YoloSeg::apply_yolo_style_nms(float iou_threshold, 
                                       float conf_threshold) {
        // 1. 过滤低置信度检测框
        this->detections_.erase(std::remove_if(this->detections_.begin(), this->detections_.end(),
            [conf_threshold](const SegResult& res) { 
                return res.confidence < conf_threshold; 
            }), this->detections_.end());
    
        // 2. 按类别分组
        std::unordered_map<int, std::vector<size_t>> class_groups;
        for (size_t i = 0; i < this->detections_.size(); ++i) {
            class_groups[this->detections_[i].class_id].push_back(i);
        }
    
        std::vector<SegResult> keep_results;
        
        // 3. 类感知NMS
        for (auto& [class_id, indices] : class_groups) {
            // 按置信度降序排序
            std::sort(indices.begin(), indices.end(), 
                [this](size_t a, size_t b) {
                    return this->detections_[a].confidence > this->detections_[b].confidence;
                });
    
            // 执行NMS
            std::vector<bool> is_suppressed(indices.size(), false);
            for (size_t i = 0; i < indices.size(); ++i) {
                if (is_suppressed[i]) continue;
    
                const auto& current_box = this->detections_[indices[i]].bbox;
                keep_results.push_back(this->detections_[indices[i]]);
    
                for (size_t j = i + 1; j < indices.size(); ++j) {
                    if (is_suppressed[j]) continue;
    
                    const auto& other_box = this->detections_[indices[j]].bbox;
                    float iou = calculate_diou(current_box, other_box);
                    if (iou > iou_threshold) {
                        // 抑制较低置信度的框
                        if (this->detections_[indices[j]].confidence < this->detections_[indices[i]].confidence) {
                            is_suppressed[j] = true;
                        }
                    }
                }
            }
        }
    
        // 4. 过滤同一类别距离过近的框
        std::vector<SegResult> final_results;
        for (size_t i = 0; i < keep_results.size(); ++i) {
            bool too_close = false;
            for (size_t j = 0; j < final_results.size(); ++j) {
                if (keep_results[i].class_id == final_results[j].class_id) {
                    float dist_x = keep_results[i].bbox.x - final_results[j].bbox.x;
                    float dist_y = keep_results[i].bbox.y - final_results[j].bbox.y;
                    float distance_2 = dist_x * dist_x + dist_y * dist_y;
                    if (distance_2 < 100.0f) { // 距离阈值，可根据需求调整
                        too_close = true;
                        break;
                    }
                }
            }
            if (!too_close) {
                final_results.push_back(keep_results[i]);
            }
        }
        this->detections_ = std::move(final_results);
    }
    
    void YoloSeg::generate_masks() {
        if (this->detections_.empty() || mask_proto_.empty()) return;
    
        // 预先将mask_protos转换为正确的形状 (32x25600)
        cv::Mat mask_protos(32, 160*160, CV_32F, mask_proto_.data);
        
        for (auto& res : this->detections_) {
            cv::Mat mask_coeffs(1, 32, CV_32F, res.mask_coeffs.data());
            cv::Mat mask(1, 160*160, CV_32F);
            
            // 矩阵乘法 (1x32) × (32x25600) = (1x25600)
            cv::gemm(mask_coeffs, mask_protos, 1.0, cv::Mat(), 0.0, mask);
            
            // 重塑为160x160，保持浮点数格式
            mask = mask.reshape(1, 160);
            // sigmoid激活
            cv::exp(-mask, mask);
            mask = 1.0 / (1.0 + mask);
            // 归一化到[0, 1]
            cv::normalize(mask, mask, 0, 1, cv::NORM_MINMAX);
            mask.convertTo(res.mask, CV_32F);
        }
    }
    
    void YoloSeg::restore_masks() {
        // Letterbox填充的上下边距（各80像素）
        const int pad_top = 80;
    
        // 遍历所有检测结果
        for (auto &res : detections_) {
            cv::Mat mask = res.mask; // 输入：160x160
            // 1. 上采样掩膜到模型输入尺寸（640x640）
            cv::Mat upsampled_mask;
            cv::resize(
                mask, 
                upsampled_mask, 
                cv::Size(input_size_, input_size_), // 目标尺寸：640x640
                0, 0, 
                cv::INTER_LINEAR // 双线性插值
            );
            // 2. 裁剪掉Letterbox填充部分（上下各80像素）
            cv::Rect roi(
                0,                          // x起始 = 0
                pad_top,                    // y起始 = 80（跳过顶部填充）
                orig_width_,             // 宽度 = 640
                orig_height_             // 高度 = 480
            );
            cv::Mat cropped_mask = upsampled_mask(roi);
            // 3. 用bbox截取有效区域（关键步骤）
            cv::Rect bbox = res.bbox; // 假设bbox格式为[x1,y1,w,h]（相对于原图480x640）
            cv::Mat bbox_mask = cv::Mat::zeros(cropped_mask.size(), cropped_mask.type());
            cropped_mask(bbox).copyTo(bbox_mask(bbox)); // 仅保留bbox内的像素
            // // 4. sigmoid激活
            // cv::exp(-bbox_mask, bbox_mask);
            // bbox_mask = 1.0 / (1.0 + bbox_mask);
            // // 5. 归一化到[0, 1]
            // cv::normalize(bbox_mask, bbox_mask, 0, 1, cv::NORM_MINMAX);
            // 6. 二值化掩膜
            cv::threshold(bbox_mask, bbox_mask, mask_thresh_, 1, cv::THRESH_BINARY);
            // 4. 转换为8位无符号整数
            bbox_mask.convertTo(bbox_mask, CV_8UC1, 255.0);
            // 形态学平滑
            // cv::Mat kernel = cv::getStructuringElement(cv::MORPH_ELLIPSE, cv::Size(3, 3));
            // cv::morphologyEx(bbox_mask, bbox_mask, cv::MORPH_CLOSE, kernel);
            // cv::morphologyEx(bbox_mask, bbox_mask, cv::MORPH_OPEN, kernel);
    
            res.mask = bbox_mask; // 更新掩膜
        }
    }
    
    float YoloSeg::calculate_diou(const cv::Rect_<float>& a, const cv::Rect_<float>& b) {
        // 1. 计算交集面积
        float inter_x1 = std::max(a.x, b.x);
        float inter_y1 = std::max(a.y, b.y);
        float inter_x2 = std::min(a.x + a.width, b.x + b.width);
        float inter_y2 = std::min(a.y + a.height, b.y + b.height);
        
        float inter_w = std::max(0.0f, inter_x2 - inter_x1);
        float inter_h = std::max(0.0f, inter_y2 - inter_y1);
        float inter_area = inter_w * inter_h;
    
        // 2. 计算并集面积
        float area_a = a.width * a.height;
        float area_b = b.width * b.height;
        float union_area = area_a + area_b - inter_area;
    
        // 3. 计算IoU（处理除零情况）
        float iou = 0.0f;
        if (union_area > 0.0f) {
            iou = inter_area / union_area;
        } else {
            return 0.0f;  // 无重叠时直接返回0
        }
    
        // 4. 计算中心点距离平方
        float center_a_x = a.x + a.width / 2;
        float center_a_y = a.y + a.height / 2;
        float center_b_x = b.x + b.width / 2;
        float center_b_y = b.y + b.height / 2;
        
        float center_dist_x = center_a_x - center_b_x;
        float center_dist_y = center_a_y - center_b_y;
        float center_dist_sq = center_dist_x * center_dist_x + center_dist_y * center_dist_y;
    
        // 5. 计算最小包围框对角线平方
        float enclose_x1 = std::min(a.x, b.x);
        float enclose_y1 = std::min(a.y, b.y);
        float enclose_x2 = std::max(a.x + a.width, b.x + b.width);
        float enclose_y2 = std::max(a.y + a.height, b.y + b.height);
        
        float enclose_w = enclose_x2 - enclose_x1;
        float enclose_h = enclose_y2 - enclose_y1;
        float enclose_diag_sq = enclose_w * enclose_w + enclose_h * enclose_h;
    
        // 6. 计算DIoU（处理除零情况）
        if (enclose_diag_sq > 0.0f) {
            return iou - (center_dist_sq / enclose_diag_sq);
        } else {
            return iou;  // 包围框对角线为0时退化为IoU
        }
    }
    
    bool YoloSeg::isNewImageArrived() {
        std::unique_lock<std::mutex> lock(mMutexGetNewImage);
        if(mbNewImageFlag)
        {
            mbNewImageFlag=false;
            return true;
        }
        else
            return false;
    }
    
    void YoloSeg::run() {
        std::cout<<"Instance segmentation thread start ..."<<std::endl;
        while (1)
        {
            usleep(1);
            if(!isNewImageArrived()) continue;
            if(mImageToSeg.empty()) continue;
            DetectAndSeg(mImageToSeg);
            ImageSegFinished();
        }
    }

    void YoloSeg::SetTracker(Tracking* tracker) {
        mpTracker = tracker;
    }

    void YoloSeg::ImageSegFinished()
    {
        std::unique_lock <std::mutex> lock(mMutexImageSegFinished);
        mpTracker->mbSegFinishedFlag=true;
    }
    float YoloSeg::sigmoid(float x){
        return 1.f / (1.f + std::exp(-x));
    }
    void YoloSeg::DrawSegmentation(cv::Mat& image) {
        std::vector<SegResult> current_detections;
        {
            std::lock_guard<std::mutex> lock(mMutex);
            current_detections = detections_to_show_; // 拷贝（避免长时间持锁）
        }
        if (current_detections.empty()) return;
        assert(image.size() == cv::Size(640, 500));
        for (const auto& detection : current_detections) {
            cv::Mat padded_mask = cv::Mat::zeros(image.size(), detection.mask.type());
            // 将原始掩膜复制到填充后掩膜的顶部（不覆盖底部填充区域）
            detection.mask.copyTo(padded_mask(cv::Rect(0, 0, 640, 480)));
            // 3. 生成掩膜颜色（仅对非零区域）
            cv::Mat mask_color = cv::Mat::zeros(image.size(), CV_8UC3);
            cv::Mat mask_8u;
            padded_mask.convertTo(mask_8u, CV_8UC1); // 转换为0~255
            cv::Vec3b color = cv::Vec3b(colors_[detection.class_id][2], 
                                        colors_[detection.class_id][1], 
                                        colors_[detection.class_id][0]);
            mask_color.setTo(color, padded_mask); // 仅对非零区域设置颜色

            // 4. 混合掩膜颜色和原图，降低掩膜颜色的透明度
            cv::Mat blended_image;
            cv::addWeighted(image, 0.5, mask_color, 0.5, 0, blended_image);
            // 5. 仅将非零掩膜区域叠加到image
            cv::Mat nonzero_region;
            cv::compare(padded_mask, 0, nonzero_region, cv::CMP_GT); // 找到mask>0的像素
            blended_image.copyTo(image, nonzero_region); // 仅修改目标区域
            
            // 6. 绘制调整后的bbox和文字
            cv::rectangle(image, detection.bbox, cv::Scalar(255, 0, 0), 1);
            std::string label = detection.class_name + " " + cv::format("%.2f", detection.confidence);
            cv::putText(image, label, cv::Point(detection.bbox.x, std::max(detection.bbox.y-5, 5.0f)),
                        cv::FONT_HERSHEY_SIMPLEX, 0.5, cv::Scalar(0, 0, 255), 2);
        }
    }
} // namespace ORB_SLAM2
