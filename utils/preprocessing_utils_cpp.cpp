#include <iostream>
#include <vector>
#include <string>
#include <algorithm>
#include <opencv2/opencv.hpp>  
#include <sndfile.h>           
#include <boost/tokenizer.hpp>  
#include <fstream>
#include <chrono>
#include <cmath>

// Utility function for logging
void log_message(const std::string &message) {
    std::ofstream log_file("preprocessing.log", std::ios_base::app);
    log_file << message << std::endl;
}

// Timer class for measuring execution time
class Timer {
    std::chrono::time_point<std::chrono::high_resolution_clock> start_time;

public:
    void start() {
        start_time = std::chrono::high_resolution_clock::now();
    }

    void stop(const std::string &task_name) {
        auto end_time = std::chrono::high_resolution_clock::now();
        std::chrono::duration<double> elapsed = end_time - start_time;
        log_message(task_name + " took " + std::to_string(elapsed.count()) + " seconds.");
    }
};

// Text Preprocessing: Tokenization, Lowercasing, and Stopword Removal
class TextPreprocessor {
public:
    std::vector<std::string> tokenize(const std::string &text) {
        std::vector<std::string> tokens;
        boost::tokenizer<> tok(text);
        for (auto it = tok.begin(); it != tok.end(); ++it) {
            tokens.push_back(*it);
        }
        return tokens;
    }

    std::string to_lowercase(const std::string &text) {
        std::string lower_text = text;
        std::transform(lower_text.begin(), lower_text.end(), lower_text.begin(), ::tolower);
        return lower_text;
    }

    std::vector<std::string> remove_stopwords(const std::vector<std::string> &tokens, const std::vector<std::string> &stopwords) {
        std::vector<std::string> filtered_tokens;
        for (const auto &token : tokens) {
            if (std::find(stopwords.begin(), stopwords.end(), token) == stopwords.end()) {
                filtered_tokens.push_back(token);
            }
        }
        return filtered_tokens;
    }

    void preprocess_text(const std::string &text, const std::vector<std::string>> windowed_signal);
        int num_windows = audio_signal.size() / window_size;
        for (int i = 0; i < num_windows; ++i) {
            std::vector<float> window(audio_signal.begin() + i * window_size, audio_signal.begin() + (i + 1) * window_size);
            windowed_signal.insert(windowed_signal.end(), window.begin(), window.end());
        }
        return windowed_signal;
    }

    std::vector<float> apply_lowpass_filter(const std::vector<float>& audio_signal, float cutoff_frequency, int sample_rate) {
        std::vector<float> filtered_signal = audio_signal;

        // Implement low-pass filter (using Butterworth filter)
        float RC = 1.0f / (2.0f * M_PI * cutoff_frequency);
        float dt = 1.0f / sample_rate;
        float alpha = dt / (RC + dt);
        
        filtered_signal[0] = audio_signal[0];
        
        for (size_t i = 1; i < audio_signal.size(); ++i) {
            filtered_signal[i] = filtered_signal[i-1] + alpha * (audio_signal[i] - filtered_signal[i-1]);
        }
        return filtered_signal;
    }

    std::vector<float> load_audio_file(const std::string& file_path) {
        SF_INFO sfinfo;
        SNDFILE* file = sf_open(file_path.c_str(), SFM_READ, &sfinfo);
        if (!file) {
            throw std::runtime_error("Error reading audio file: " + file_path);
        }

        std::vector<float> audio_signal(sfinfo.frames * sfinfo.channels);
        sf_read_float(file, audio_signal.data(), sfinfo.frames * sfinfo.channels);
        sf_close(file);
        return audio_signal;
    }

// Utility class to demonstrate full pipeline processing
class MultimodalPreprocessor {
private:
    TextPreprocessor text_processor;
    ImagePreprocessor image_processor;
    AudioPreprocessor audio_processor;

public:
    // Complete text processing pipeline
    void process_text(const std::string& text, const std::vector<std::string>& stopwords) {
        std::string clean_text = Utils::remove_punctuation(text);
        std::string lower_text = text_processor.to_lowercase(clean_text);
        std::vector<std::string> tokens = text_processor.tokenize(lower_text);
        std::vector<std::string> filtered_tokens = text_processor.remove_stopwords(tokens, stopwords);
        std::vector<std::string> stemmed_tokens = text_processor.stem_tokens(filtered_tokens);

        std::cout << "Processed Text Tokens: ";
        for (const auto& token : stemmed_tokens) {
            std::cout << token << " ";
        }
        std::cout << std::endl;
    }

    // Complete image processing pipeline
    void process_image(const std::string& image_path) {
        cv::Mat image = cv::imread(image_path);
        if (image.empty()) {
            throw std::runtime_error("Error reading image file: " + image_path);
        }

        cv::Mat resized_image = image_processor.resize_image(image, 256, 256);
        cv::Mat grayscale_image = image_processor.convert_to_grayscale(resized_image);
        cv::Mat equalized_image = image_processor.histogram_equalization(grayscale_image);
        cv::Mat normalized_image = image_processor.normalize_image(equalized_image);

        std::cout << "Processed Image: Width = " << normalized_image.cols << ", Height = " << normalized_image.rows << std::endl;
    }

    // Complete audio processing pipeline
    void process_audio(const std::string& audio_path) {
        std::vector<float> audio_signal = audio_processor.load_audio_file(audio_path);
        std::vector<float> denoised_signal = audio_processor.reduce_noise(audio_signal);
        std::vector<float> windowed_signal = audio_processor.apply_windowing(denoised_signal, 512);
        std::vector<float> spectrogram = audio_processor.compute_spectrogram(windowed_signal, 16000, 512, 256);

        std::cout << "Processed Audio Spectrogram with " << spectrogram.size() << " elements." << std::endl;
    }
};

int main() {
    // Instantiate the multimodal preprocessor
    MultimodalPreprocessor preprocessor;

    // Text processing
    std::string text = "Running advanced text preprocessing tasks, removing punctuation, and stemming words.";
    std::vector<std::string> stopwords = {"and", "the", "is", "to"};
    preprocessor.process_text(text, stopwords);

    // Image processing
    preprocessor.process_image("sample_image.jpg");

    // Audio processing
    preprocessor.process_audio("sample_audio.wav");

    return 0;
}