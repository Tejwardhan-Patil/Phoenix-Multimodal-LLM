#include <iostream>
#include <vector>
#include <cmath>
#include <fftw3.h>
#include <stdexcept>
#include <fstream>

// Function to read audio data from a file
std::vector<double> read_audio_data(const std::string& file_path) {
    std::vector<double> audio_data;
    std::ifstream file(file_path);
    double sample;

    if (!file.is_open()) {
        throw std::runtime_error("Error: Unable to open audio file.");
    }

    while (file >> sample) {
        audio_data.push_back(sample);
    }

    file.close();
    return audio_data;
}

// Function to normalize audio data to a specific range [-1, 1]
void normalize_audio(std::vector<double>& audio_data) {
    double max_value = 0.0;

    // Find maximum absolute value
    for (const auto& sample : audio_data) {
        if (std::abs(sample) > max_value) {
            max_value = std::abs(sample);
        }
    }

    // Normalize audio data
    if (max_value > 0.0) {
        for (auto& sample : audio_data) {
            sample /= max_value;
        }
    }
}

// Function to apply a Hann window
void apply_hann_window(std::vector<double>& window) {
    int size = window.size();
    for (int i = 0; i < size; ++i) {
        window[i] *= 0.5 * (1 - cos(2 * M_PI * i / (size - 1)));
    }
}

// Function to apply a simple noise filter
void apply_noise_filter(std::vector<double>& audio_data, double threshold) {
    for (auto& sample : audio_data) {
        if (std::abs(sample) < threshold) {
            sample = 0.0;
        }
    }
}

// Function to compute and print the spectrogram
void compute_spectrogram(const std::vector<double>& audio_data, int sample_rate, int window_size) {
    int num_windows = audio_data.size() / window_size;
    fftw_complex *input, *output;
    fftw_plan plan;

    // Allocate memory for FFTW input/output
    input = (fftw_complex*) fftw_malloc(sizeof(fftw_complex) * window_size);
    output = (fftw_complex*) fftw_malloc(sizeof(fftw_complex) * window_size);

    // Vector to store windowed data
    std::vector<double> window(window_size);

    // Loop through each window of the audio data
    for (int i = 0; i < num_windows; ++i) {
        // Copy and apply window function
        for (int j = 0; j < window_size; ++j) {
            window[j] = audio_data[i * window_size + j];
        }

        apply_hann_window(window);

        // Copy windowed data to FFTW input
        for (int j = 0; j < window_size; ++j) {
            input[j][0] = window[j]; // Real part
            input[j][1] = 0.0;       // Imaginary part
        }

        // Create FFTW plan and execute it
        plan = fftw_plan_dft_1d(window_size, input, output, FFTW_FORWARD, FFTW_ESTIMATE);
        fftw_execute(plan);

        // Print the FFT result (compute magnitude)
        std::cout << "Spectrogram for window " << i << ":\n";
        for (int k = 0; k < window_size; ++k) {
            double magnitude = sqrt(output[k][0] * output[k][0] + output[k][1] * output[k][1]);
            std::cout << magnitude << " ";
        }
        std::cout << std::endl;

        // Clean up FFTW plan
        fftw_destroy_plan(plan);
    }

    // Free memory
    fftw_free(input);
    fftw_free(output);
}

// Function to save spectrogram data to a file (for further analysis or visualization)
void save_spectrogram_to_file(const std::vector<std::vector<double>>& spectrogram, const std::string& file_path) {
    std::ofstream file(file_path);

    if (!file.is_open()) {
        throw std::runtime_error("Error: Unable to open file for saving spectrogram.");
    }

    for (const auto& window : spectrogram) {
        for (const auto& value : window) {
            file << value << " ";
        }
        file << "\n";
    }

    file.close();
}

// Main preprocessing pipeline
void preprocess_audio(const std::string& file_path, double noise_threshold, int sample_rate, int window_size) {
    try {
        // Step 1: Read the audio data
        std::vector<double> audio_data = read_audio_data(file_path);

        // Step 2: Normalize the audio data
        normalize_audio(audio_data);

        // Step 3: Apply noise filter
        apply_noise_filter(audio_data, noise_threshold);

        // Step 4: Compute spectrogram
        compute_spectrogram(audio_data, sample_rate, window_size);
    }
    catch (const std::runtime_error& e) {
        std::cerr << e.what() << std::endl;
    }
}

// Main entry point of the program
int main(int argc, char* argv[]) {
    if (argc != 5) {
        std::cerr << "Usage: " << argv[0] << " <audio_file> <noise_threshold> <sample_rate> <window_size>" << std::endl;
        return 1;
    }

    std::string file_path = argv[1];
    double noise_threshold = std::stod(argv[2]);
    int sample_rate = std::stoi(argv[3]);
    int window_size = std::stoi(argv[4]);

    // Preprocess the audio data
    preprocess_audio(file_path, noise_threshold, sample_rate, window_size);

    return 0;
}