#include "yolo.hpp"

YoloNode::YoloNode() : Node("yolo_node"), is_running_(true) {
    // Declare and get parameters
    this->declare_parameter("headless", false);
    this->declare_parameter("hitl", false);
    this->declare_parameter("hfov", 90.0);
    this->declare_parameter("vfov", 60.0);

    headless_ = this->get_parameter("headless").as_bool();
    hitl_ = this->get_parameter("hitl").as_bool();
    hfov_ = this->get_parameter("hfov").as_double();
    vfov_ = this->get_parameter("vfov").as_double();

    // Detect Architecture
    struct utsname buffer;
    if (uname(&buffer) == 0) {
        architecture_ = std::string(buffer.machine);
    } else {
        architecture_ = "unknown";
    }
    RCLCPP_INFO(this->get_logger(), "Detected Architecture: %s", architecture_.c_str());

    // Load Classes
    load_classes("/aas/yolo/coco.json");
    
    // Generate random colors
    cv::RNG rng(12345);
    for(size_t i=0; i<class_names_.size(); i++) {
        colors_.emplace_back(rng.uniform(0, 255), rng.uniform(0, 255), rng.uniform(0, 255));
    }

    // Load Model
    load_model("/aas/yolo/yolov8n.onnx");

    // Create Publisher
    detection_publisher_ = this->create_publisher<vision_msgs::msg::Detection2DArray>("detections", 10);

    RCLCPP_INFO(this->get_logger(), "YOLO inference started.");

    // Start threads
    std::string pipeline = get_gstreamer_pipeline();
    frame_thread_ = std::thread(&YoloNode::frame_capture_thread, this, pipeline);
    inference_thread_ = std::thread(&YoloNode::run_inference_loop, this);
}

YoloNode::~YoloNode() {
    is_running_ = false;
    if (frame_thread_.joinable()) frame_thread_.join();
    if (inference_thread_.joinable()) inference_thread_.join();
    if (!headless_) cv::destroyAllWindows();
}

void YoloNode::load_classes(const std::string& path) {
    // Simple manual JSON parsing (assuming {"0": "person", ...} format), alternatively, use nlohmann/json
    std::ifstream file(path);
    if (!file.is_open()) {
        RCLCPP_WARN(this->get_logger(), "Could not open classes file. Using defaults.");
        for(int i=0; i<80; i++) class_names_.push_back(std::to_string(i));
        return;
    }

    std::string line;
    std::string content;
    while(std::getline(file, line)) content += line;

    // REGEX parsing to avoid extra dependencies
    std::regex re("\"(\\d+)\":\\s*\"([^\"]+)\"");
    std::sregex_iterator next(content.begin(), content.end(), re);
    std::sregex_iterator end;
    
    std::map<int, std::string> map_classes;
    while (next != end) {
        std::smatch match = *next;
        map_classes[std::stoi(match.str(1))] = match.str(2);
        next++;
    }

    for(auto const& [key, val] : map_classes) {
        if(key >= (int)class_names_.size()) class_names_.resize(key + 1);
        class_names_[key] = val;
    }
}

void YoloNode::load_model(const std::string& path) {
    env_ = std::make_shared<Ort::Env>(ORT_LOGGING_LEVEL_WARNING, "YoloNode");
    Ort::SessionOptions session_options;
    
    // Try to append CUDA execution provider
    try {
        if (architecture_ == "x86_64" || architecture_ == "aarch64") {
            // Note: TODO TensorRT, cache, etc.
            OrtCUDAProviderOptions cuda_options;
            session_options.AppendExecutionProvider_CUDA(cuda_options);
            RCLCPP_INFO(this->get_logger(), "Attempting to use CUDA Execution Provider.");
        }
    } catch (...) {
        RCLCPP_WARN(this->get_logger(), "CUDA Provider failed, falling back to CPU.");
    }
    
    // Create session
    try {
        session_ = std::make_shared<Ort::Session>(*env_, path.c_str(), session_options);
    } catch (const std::exception& e) {
        RCLCPP_ERROR(this->get_logger(), "Failed to load model: %s", e.what());
        throw;
    }

    // Get input name
    Ort::AllocatorWithDefaultOptions allocator;
    auto input_name_ptr = session_->GetInputNameAllocated(0, allocator);
    input_name_ = std::string(input_name_ptr.get());
}

std::string YoloNode::get_gstreamer_pipeline() {
    if (architecture_ == "x86_64") {
        return "udpsrc port=5600 ! "
               "application/x-rtp, media=(string)video, encoding-name=(string)H264 ! "
               "rtph264depay ! "
               "avdec_h264 ! " // removed option: threads=4
               "videoconvert ! "
               "video/x-raw, format=BGR ! appsink"; // tried removing: video/x-raw, format=BGR
               // Test with:
               // GST_DEBUG=3 gst-launch-1.0 udpsrc port=5600 ! application/x-rtp, media=video, encoding-name=H264 ! rtph264depay ! avdec_h264 ! videoconvert ! autovideosink 
               // Python fallback: ros2 run yolo_inference_py yolo_inference_py --ros-args -p use_sim_time:=true
    } else if (architecture_ == "aarch64") {
        if (hitl_) {
            return "udpsrc port=5600 ! "
                   "application/x-rtp, media=(string)video, encoding-name=(string)H264 ! "
                   "rtph264depay ! "
                   "h264parse ! "
                   "nvv4l2decoder ! "
                   "nvvidconv ! "
                   "video/x-raw, format=I420 ! "
                   "videoconvert ! "
                   "video/x-raw, format=BGR ! "
                   "appsink drop=true max-buffers=1";
        } else {
            return "nvarguscamerasrc sensor-id=0 ! "
                   "video/x-raw(memory:NVMM), width=1280, height=720, framerate=60/1 ! "
                   "nvvidconv ! "
                   "video/x-raw, format=BGRx, width=1280, height=720, framerate=60/1 ! "
                   "videoconvert ! "
                   "appsink drop=true max-buffers=1 sync=false";
        }
    }
    return ""; // Default fail
}

void YoloNode::frame_capture_thread(std::string pipeline) {
    cv::VideoCapture cap(pipeline, cv::CAP_GSTREAMER);
    if (!cap.isOpened()) {
        RCLCPP_ERROR(this->get_logger(), "Failed to open video stream.");
        return;
    }

    cv::Mat frame;
    int frame_count = 0;
    auto start_time = std::chrono::steady_clock::now();

    while (is_running_ && rclcpp::ok()) {
        if (!cap.read(frame)) {
            std::this_thread::sleep_for(std::chrono::milliseconds(10));
            continue;
        }

        {
            std::lock_guard<std::mutex> lock(queue_mutex_);
            if (frame_queue_.size() >= max_queue_size_) {
                frame_queue_.pop(); // Drop oldest
            }
            frame_queue_.push(frame.clone());
        }
        
        frame_count++;
        if (frame_count % 120 == 0) {
             auto end_time = std::chrono::steady_clock::now();
             double elapsed = std::chrono::duration_cast<std::chrono::milliseconds>(end_time - start_time).count() / 1000.0;
             RCLCPP_INFO(this->get_logger(), "Frame Rx Rate: %.2f FPS", frame_count / elapsed);
             frame_count = 0;
             start_time = std::chrono::steady_clock::now();
        }
    }
    cap.release();
}

void YoloNode::run_inference_loop() {
    if (!headless_) {
        std::string win_name = "YOLOv8";
        cv::namedWindow(win_name, cv::WINDOW_NORMAL);
    }

    cv::Mat frame;
    int inf_count = 0;
    auto start_time = std::chrono::steady_clock::now();

    while (is_running_ && rclcpp::ok()) {
        bool has_frame = false;
        {
            std::lock_guard<std::mutex> lock(queue_mutex_);
            if (!frame_queue_.empty()) {
                frame = frame_queue_.front();
                frame_queue_.pop();
                has_frame = true;
            }
        }

        if (!has_frame) {
            std::this_thread::sleep_for(std::chrono::milliseconds(5));
            continue;
        }

        auto detections = run_yolo(frame);
        publish_detections(frame, detections);

        inf_count++;
        if (inf_count % 120 == 0) {
            auto end_time = std::chrono::steady_clock::now();
            double elapsed = std::chrono::duration_cast<std::chrono::milliseconds>(end_time - start_time).count() / 1000.0;
            RCLCPP_INFO(this->get_logger(), "Inference Rate: %.2f FPS", inf_count / elapsed);
            inf_count = 0;
            start_time = std::chrono::steady_clock::now();
        }

        if (!headless_) {
            visualize(frame, detections);
            cv::waitKey(1);
        }
    }
}

std::vector<YoloNode::Detection> YoloNode::run_yolo(const cv::Mat& frame) {
    // Preprocess
    int input_w = 640;
    int input_h = 640;
    cv::Mat blob;
    cv::dnn::blobFromImage(
        frame,                                       // Arg 1: InputArray image
        blob,                                        // Arg 2: OutputArray blob (the fix!)
        1.0/255.0,                                   // Arg 3: double scalefactor
        cv::Size(input_w, input_h),                  // Arg 4: const Size& size
        cv::Scalar(),                                // Arg 5: const Scalar& mean
        true,                                        // Arg 6: bool swapRB
        false                                        // Arg 7: bool crop
    ); 
    // The 8th argument (CV_32F) is typically implied by the OutputArray type in this signature.

    // Setup Inputs
    Ort::MemoryInfo memory_info = Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault);
    std::vector<int64_t> input_shape = {1, 3, input_h, input_w};
    size_t input_tensor_size = 1 * 3 * input_h * input_w;
    
    // Create input tensor (data, count, shape, shape_len)
    Ort::Value input_tensor = Ort::Value::CreateTensor<float>(
        memory_info, blob.ptr<float>(), input_tensor_size, input_shape.data(), input_shape.size());

    // Run Inference
    const char* input_names[] = {input_name_.c_str()};
    const char* output_names[] = {"output0"};
    
    auto output_tensors = session_->Run(
        Ort::RunOptions{nullptr}, input_names, &input_tensor, 1, output_names, 1);

    // Post Process
    float* output_data = output_tensors[0].GetTensorMutableData<float>();
    
    // Output shape is [1, 84, 8400] -> (1, 4+80, anchors)
    // int dimensions = 84; // 4 box + 80 classes
    int anchors = 8400;

    std::vector<int> class_ids;
    std::vector<float> confidences;
    std::vector<cv::Rect> boxes;

    float x_scale = (float)frame.cols / input_w;
    float y_scale = (float)frame.rows / input_h;

    // The output is (1, 84, 8400). Transpose logic.
    // Data is stored flat. index = (dim * anchors) + anchor_index
    
    for (int i = 0; i < anchors; ++i) {
        // Find max class score
        float max_score = 0.0f;
        int max_class_id = -1;
        
        // Classes start at index 4 (0-3 are x,y,w,h)
        for (int c = 0; c < 80; ++c) {
            float score = output_data[(4 + c) * anchors + i];
            if (score > max_score) {
                max_score = score;
                max_class_id = c;
            }
        }

        if (max_score > 0.5f) {
            float cx = output_data[0 * anchors + i];
            float cy = output_data[1 * anchors + i];
            float w = output_data[2 * anchors + i];
            float h = output_data[3 * anchors + i];

            int left = int((cx - 0.5 * w) * x_scale);
            int top = int((cy - 0.5 * h) * y_scale);
            int width = int(w * x_scale);
            int height = int(h * y_scale);

            boxes.push_back(cv::Rect(left, top, width, height));
            confidences.push_back(max_score);
            class_ids.push_back(max_class_id);
        }
    }

    // NMS
    std::vector<int> indices;
    cv::dnn::NMSBoxes(boxes, confidences, 0.5f, 0.45f, indices);

    std::vector<Detection> final_detections;
    for (int idx : indices) {
        Detection det;
        det.class_id = class_ids[idx];
        det.confidence = confidences[idx];
        det.box = boxes[idx];
        final_detections.push_back(det);
    }
    return final_detections;
}

void YoloNode::publish_detections(const cv::Mat& frame, const std::vector<Detection>& detections) {
    vision_msgs::msg::Detection2DArray msg;
    msg.header.stamp = this->get_clock()->now();
    msg.header.frame_id = "camera_frame";

    float image_width = (float)frame.cols;
    float image_height = (float)frame.rows;

    for (const auto& det : detections) {
        vision_msgs::msg::Detection2D detection_msg;
        
        // BBox
        detection_msg.bbox.center.position.x = det.box.x + det.box.width / 2.0;
        detection_msg.bbox.center.position.y = det.box.y + det.box.height / 2.0;
        detection_msg.bbox.size_x = det.box.width;
        detection_msg.bbox.size_y = det.box.height;

        // Hypothesis
        vision_msgs::msg::ObjectHypothesisWithPose result;
        if (det.class_id < (int)class_names_.size())
            result.hypothesis.class_id = class_names_[det.class_id];
        else
            result.hypothesis.class_id = std::to_string(det.class_id);
            
        result.hypothesis.score = det.confidence;

        // Calculate Angle (FOV math)
        float offset_x = detection_msg.bbox.center.position.x - (image_width / 2.0);
        float offset_y = (image_height / 2.0) - detection_msg.bbox.center.position.y;
        
        float norm_x = offset_x / image_width;
        float norm_y = offset_y / image_height;
        
        result.pose.pose.position.x = norm_x * hfov_; // azimuth
        result.pose.pose.position.y = norm_y * vfov_; // elevation
        
        detection_msg.results.push_back(result);
        detection_msg.id = result.hypothesis.class_id;
        
        msg.detections.push_back(detection_msg);
    }
    detection_publisher_->publish(msg);
}

void YoloNode::visualize(cv::Mat& frame, const std::vector<Detection>& detections) {
    for (const auto& det : detections) {
        cv::Scalar color = colors_[det.class_id % colors_.size()];
        cv::rectangle(frame, det.box, color, 2);
        
        std::string label = (det.class_id < (int)class_names_.size() ? class_names_[det.class_id] : "Unknown") + 
                            " " + std::to_string(det.confidence).substr(0, 4);
        
        cv::putText(frame, label, cv::Point(det.box.x, det.box.y - 5), cv::FONT_HERSHEY_SIMPLEX, 0.6, color, 1);
    }
    cv::imshow("YOLOv8", frame);
}

int main(int argc, char** argv) {
    rclcpp::init(argc, argv);
    auto node = std::make_shared<YoloNode>();
    rclcpp::spin(node);
    rclcpp::shutdown();
    return 0;
}
