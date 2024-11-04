#include <iostream>
#include <fstream>
#include <vector>
#include <string>
#include <sstream>
#include <filesystem>
#include <algorithm>
#include <locale>
#include <chrono>

// Utility function to split a string by a delimiter
std::vector<std::string> split(const std::string &line, char delimiter) {
    std::vector<std::string> tokens;
    std::stringstream ss(line);
    std::string token;
    while (std::getline(ss, token, delimiter)) {
        tokens.push_back(token);
    }
    return tokens;
}

// Function to load a CSV file and store the data in a 2D vector
std::vector<std::vector<std::string>> load_csv(const std::string &file_path) {
    std::ifstream file(file_path);
    std::vector<std::vector<std::string>> data;
    std::string line;

    if (file.is_open()) {
        while (std::getline(file, line)) {
            data.push_back(split(line, ','));
        }
        file.close();
    } else {
        std::cerr << "Could not open file: " << file_path << std::endl;
    }

    return data;
}

// Function to load all text files from a directory
std::vector<std::string> load_text_files(const std::string &directory_path) {
    std::vector<std::string> texts;

    for (const auto &entry : std::filesystem::directory_iterator(directory_path)) {
        std::ifstream file(entry.path());
        std::stringstream buffer;
        buffer << file.rdbuf();
        texts.push_back(buffer.str());
    }

    return texts;
}

// Function to load a directory of image files
std::vector<std::string> load_images(const std::string &directory_path) {
    std::vector<std::string> image_files;

    for (const auto &entry : std::filesystem::directory_iterator(directory_path)) {
        image_files.push_back(entry.path().string());
    }

    return image_files;
}

// Function to preprocess text data (lowercasing, removing punctuation)
std::string clean_text(const std::string &text) {
    std::string clean_text;
    std::locale loc;
    for (char c : text) {
        if (std::isalpha(c, loc) || std::isspace(c, loc)) {
            clean_text += std::tolower(c, loc);
        }
    }
    return clean_text;
}

// Function to batch preprocess a vector of texts
std::vector<std::string> preprocess_texts(const std::vector<std::string> &texts) {
    std::vector<std::string> preprocessed_texts;
    for (const auto &text : texts) {
        preprocessed_texts.push_back(clean_text(text));
    }
    return preprocessed_texts;
}

// Timer class to measure performance
class Timer {
public:
    Timer() : start(std::chrono::high_resolution_clock::now()) {}

    void stop() {
        auto end = std::chrono::high_resolution_clock::now();
        std::chrono::duration<float> duration = end - start;
        std::cout << "Duration: " << duration.count() << "s" << std::endl;
    }

private:
    std::chrono::high_resolution_clock::time_point start;
};

// Function to display CSV data
void display_csv(const std::vector<std::vector<std::string>> &csv_data) {
    std::cout << "CSV Data:" << std::endl;
    for (const auto &row : csv_data) {
        for (const auto &cell : row) {
            std::cout << cell << " ";
        }
        std::cout << std::endl;
    }
}

// Function to display image file paths
void display_images(const std::vector<std::string> &images) {
    std::cout << "Loaded Images:" << std::endl;
    for (const auto &image : images) {
        std::cout << image << std::endl;
    }
}

// Function to display preprocessed text
void display_preprocessed_texts(const std::vector<std::string> &texts) {
    std::cout << "Preprocessed Texts:" << std::endl;
    for (const auto &text : texts) {
        std::cout << text << std::endl;
    }
}

// Function to save preprocessed text data to a file
void save_preprocessed_texts(const std::vector<std::string> &texts, const std::string &file_path) {
    std::ofstream outfile(file_path);
    if (outfile.is_open()) {
        for (const auto &text : texts) {
            outfile << text << std::endl;
        }
        outfile.close();
        std::cout << "Preprocessed texts saved to " << file_path << std::endl;
    } else {
        std::cerr << "Unable to open file for writing: " << file_path << std::endl;
    }
}

// Function to calculate the number of words in a text
size_t count_words(const std::string &text) {
    std::istringstream iss(text);
    return std::distance(std::istream_iterator<std::string>(iss), std::istream_iterator<std::string>());
}

// Function to count total words in a batch of texts
size_t count_total_words(const std::vector<std::string> &texts) {
    size_t total_words = 0;
    for (const auto &text : texts) {
        total_words += count_words(text);
    }
    return total_words;
}

// Loading text, images, and CSV data, measuring performance
int main() {
    Timer timer;

    // Load CSV data
    std::string csv_file = "data/dataset.csv";
    auto csv_data = load_csv(csv_file);
    display_csv(csv_data);

    // Load and preprocess text files
    std::string text_dir = "data/texts";
    auto texts = load_text_files(text_dir);
    auto preprocessed_texts = preprocess_texts(texts);
    display_preprocessed_texts(preprocessed_texts);
    
    // Count total words in preprocessed texts
    size_t total_words = count_total_words(preprocessed_texts);
    std::cout << "Total words in preprocessed texts: " << total_words << std::endl;

    // Save preprocessed texts
    std::string output_file = "data/preprocessed_texts.txt";
    save_preprocessed_texts(preprocessed_texts, output_file);

    // Load image files
    std::string image_dir = "data/images";
    auto images = load_images(image_dir);
    display_images(images);

    timer.stop();
    return 0;
}