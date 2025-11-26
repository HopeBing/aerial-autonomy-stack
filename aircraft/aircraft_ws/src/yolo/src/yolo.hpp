#ifndef YOLO_HPP_
#define YOLO_HPP_

#include <rclcpp/rclcpp.hpp>
#include <vision_msgs/msg/detection2_d_array.hpp>
#include <vision_msgs/msg/detection2_d.hpp>
#include <sensor_msgs/msg/image.hpp>
#include <cv_bridge/cv_bridge.h>

#include <opencv2/opencv.hpp>
#include <opencv2/dnn.hpp>
#include <onnxruntime_cxx_api.h>

#include <thread>
#include <mutex>
#include <queue>
#include <atomic>
#include <map>
#include <string>

#include <sys/utsname.h>
#include <fstream>
#include <iostream>
#include <regex>

class YoloNode : public rclcpp::Node {
public:
    YoloNode();
    ~YoloNode();

    void run_inference_loop();

private:
    // Config parameters
    bool headless_;
    bool hitl_;
    float hfov_;
    float vfov_;
    std::string architecture_;

    // ROS
    rclcpp::Publisher<vision_msgs::msg::Detection2DArray>::SharedPtr detection_publisher_;
    
    // ONNX Runtime
    std::shared_ptr<Ort::Env> env_;
    std::shared_ptr<Ort::Session> session_;
    std::string input_name_;
    std::vector<std::string> class_names_;
    std::vector<cv::Scalar> colors_;

    // Threading
    std::thread frame_thread_;
    std::thread inference_thread_;
    std::atomic<bool> is_running_;
    
    // Frame Buffer
    std::queue<cv::Mat> frame_queue_;
    std::mutex queue_mutex_;
    const size_t max_queue_size_ = 3;

    // Helper functions
    void frame_capture_thread(std::string pipeline);
    void load_classes(const std::string& path);
    void load_model(const std::string& path);
    std::string get_gstreamer_pipeline();
    
    // Inference
    struct Detection {
        int class_id;
        float confidence;
        cv::Rect box;
    };
    
    std::vector<Detection> run_yolo(const cv::Mat& frame);
    void publish_detections(const cv::Mat& frame, const std::vector<Detection>& detections);
    void visualize(cv::Mat& frame, const std::vector<Detection>& detections);
};

#endif // YOLO_HPP_
