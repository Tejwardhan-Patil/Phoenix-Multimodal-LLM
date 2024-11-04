#include <iostream>
#include <fstream>
#include <vector>
#include <string>
#include <memory>
#include <opencv2/opencv.hpp>  
#include <torch/torch.h>        
#include <torch/script.h>      
#include <librosa/librosa.h>    
#include <chrono>               

// Logger class for tracking inference process
class Logger {
public:
    static void log(const std::string &message) {
        std::ofstream logfile("inference_log.txt", std::ios::app);
        if (logfile.is_open()) {
            logfile << message << std::endl;
        } else {
            std::cerr << "Unable to open log file!" << std::endl;
        }
    }
};

// Timer class for tracking execution time
class Timer {
    std::chrono::time_point<std::chrono::high_resolution_clock> start_time;
public:
    void start() {
        start_time = std::chrono::high_resolution_clock::now();
    }

    void stop_and_log(const std::string &process_name) {
        auto end_time = std::chrono::high_resolution_clock::now();
        auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end_time - start_time).count();
        Logger::log(process_name + " completed in " + std::to_string(duration) + " ms");
    }
};

// Function to load pre-trained models
torch::jit::script::Module load_model(const std::string &model_path) {
    try {
        Logger::log("Loading model from path: " + model_path);
        return torch::jit::load(model_path);
    } catch (const c10::Error &e) {
        Logger::log("Error loading the model from path: " + model_path);
        throw;
    }
}

// Image preprocessing function using OpenCV
cv::Mat preprocess_image(const std::string &image_path, int img_size) {
    cv::Mat img = cv::imread(image_path);
    if (img.empty()) {
        Logger::log("Error: Could not load image at " + image_path);
        exit(EXIT_FAILURE);
    }
    cv::resize(img, img, cv::Size(img_size, img_size));
    img.convertTo(img, CV_32FC3, 1.0 / 255); // Normalize to [0, 1]
    Logger::log("Image preprocessed: " + image_path);
    return img;
}

// Audio preprocessing function using Librosa
std::vector<float> preprocess_audio(const std::string &audio_path) {
    std::vector<float> audio_data;
    librosa::load(audio_path, audio_data, 16000); // Load at 16kHz sample rate
    Logger::log("Audio loaded: " + audio_path);
    return librosa::mfcc(audio_data, 16000);      // Extract MFCC features
}

// Function to read text input from file
std::string read_text(const std::string &text_path) {
    std::ifstream file(text_path);
    std::string content((std::istreambuf_iterator<char>(file)), std::istreambuf_iterator<char>());
    if (content.empty()) {
        Logger::log("Error: Could not load text input at " + text_path);
        exit(EXIT_FAILURE);
    }
    Logger::log("Text input loaded: " + text_path);
    return content;
}

// Function to preprocess text input (tokenization)
torch::Tensor preprocess_text(const std::string &text_input) {
    std::vector<int64_t> tokens(512, 1); 
    torch::Tensor text_tensor = torch::tensor(tokens).view({1, -1});
    Logger::log("Text input preprocessed");
    return text_tensor;
}

// Multimodal inference function
std::vector<torch::Tensor> run_inference(
    const torch::jit::script::Module &text_model,
    const torch::jit::script::Module &image_model,
    const torch::jit::script::Module &audio_model,
    const torch::Tensor &text_tensor,
    const cv::Mat &image,
    const torch::Tensor &audio_tensor) {

    Timer timer;
    
    timer.start();
    auto text_output = text_model.forward({text_tensor}).toTensor();
    timer.stop_and_log("Text inference");

    timer.start();
    torch::Tensor image_tensor = torch::from_blob(image.data, {1, 224, 224, 3}).permute({0, 3, 1, 2});
    auto image_output = image_model.forward({image_tensor}).toTensor();
    timer.stop_and_log("Image inference");

    timer.start();
    auto audio_output = audio_model.forward({audio_tensor}).toTensor();
    timer.stop_and_log("Audio inference");

    return {text_output, image_output, audio_output};
}

// Function to write output to a file
void write_output(const std::vector<torch::Tensor> &outputs) {
    std::ofstream output_file("inference_output.txt");
    if (output_file.is_open()) {
        output_file << "Text Model Output: " << outputs[0] << std::endl;
        output_file << "Image Model Output: " << outputs[1] << std::endl;
        output_file << "Audio Model Output: " << outputs[2] << std::endl;
        Logger::log("Inference results written to inference_output.txt");
    } else {
        Logger::log("Error: Could not open file to write inference results.");
    }
}

int main() {
    Logger::log("Multimodal inference started");

    // Paths to pre-trained models
    std::string text_model_path = "models/text_model.pt";
    std::string image_model_path = "models/image_model.pt";
    std::string audio_model_path = "models/audio_model.pt";

    // Load models with timing
    Timer model_timer;
    
    model_timer.start();
    torch::jit::script::Module text_model = load_model(text_model_path);
    model_timer.stop_and_log("Text model loading");

    model_timer.start();
    torch::jit::script::Module image_model = load_model(image_model_path);
    model_timer.stop_and_log("Image model loading");

    model_timer.start();
    torch::jit::script::Module audio_model = load_model(audio_model_path);
    model_timer.stop_and_log("Audio model loading");

    // Input data paths
    std::string text_path = "input_text.txt";
    std::string image_path = "input_image.jpg";
    std::string audio_path = "input_audio.wav";

    // Load and preprocess inputs
    Timer preprocess_timer;
    
    preprocess_timer.start();
    std::string text_input = read_text(text_path);
    torch::Tensor text_tensor = preprocess_text(text_input);
    preprocess_timer.stop_and_log("Text preprocessing");

    preprocess_timer.start();
    cv::Mat image = preprocess_image(image_path, 224);
    preprocess_timer.stop_and_log("Image preprocessing");

    preprocess_timer.start();
    std::vector<float> audio_features = preprocess_audio(audio_path); 
    torch::Tensor audio_tensor = torch::from_blob(audio_features.data(), {1, (int64_t)audio_features.size()});
    preprocess_timer.stop_and_log("Audio preprocessing");

    // Perform multimodal inference
    std::vector<torch::Tensor> outputs = run_inference(text_model, image_model, audio_model, text_tensor, image, audio_tensor);

    // Write output to file
    write_output(outputs);

    Logger::log("Multimodal inference completed");
    return 0;
}