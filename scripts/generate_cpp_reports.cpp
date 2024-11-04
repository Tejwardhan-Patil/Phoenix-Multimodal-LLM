#include <iostream>
#include <fstream>
#include <vector>
#include <string>
#include <chrono>
#include <sstream>
#include <iomanip>
#include <nlohmann/json.hpp> 

using json = nlohmann::json;

// Struct for storing evaluation results
struct EvaluationResult {
    std::string modality;
    double accuracy;
    double f1_score;
    double inference_time;
};

// Helper function to format double values to 2 decimal places
std::string formatDouble(double value) {
    std::ostringstream stream;
    stream << std::fixed << std::setprecision(2) << value;
    return stream.str();
}

// Function to generate a CSV report
void generateCSVReport(const std::vector<EvaluationResult>& results, const std::string& output_file) {
    std::ofstream file(output_file);
    if (file.is_open()) {
        file << "Modality,Accuracy,F1-Score,Inference Time (ms)\n";
        for (const auto& result : results) {
            file << result.modality << "," 
                 << formatDouble(result.accuracy) << "," 
                 << formatDouble(result.f1_score) << "," 
                 << formatDouble(result.inference_time) << "\n";
        }
        file.close();
        std::cout << "CSV report generated: " << output_file << std::endl;
    } else {
        std::cerr << "Error: Unable to open file: " << output_file << std::endl;
    }
}

// Function to generate a detailed text report
void generateTextReport(const std::vector<EvaluationResult>& results, const std::string& output_file) {
    std::ofstream file(output_file);
    if (file.is_open()) {
        file << "=============================\n";
        file << "      Multimodal Report\n";
        file << "=============================\n\n";
        for (const auto& result : results) {
            file << "Modality: " << result.modality << "\n";
            file << "Accuracy: " << formatDouble(result.accuracy) << "\n";
            file << "F1-Score: " << formatDouble(result.f1_score) << "\n";
            file << "Inference Time: " << formatDouble(result.inference_time) << " ms\n";
            file << "-----------------------------\n";
        }
        file.close();
        std::cout << "Text report generated: " << output_file << std::endl;
    } else {
        std::cerr << "Error: Unable to open file: " << output_file << std::endl;
    }
}

// Function to generate a JSON report
void generateJSONReport(const std::vector<EvaluationResult>& results, const std::string& output_file) {
    json j;
    for (const auto& result : results) {
        j["results"].push_back({
            {"modality", result.modality},
            {"accuracy", formatDouble(result.accuracy)},
            {"f1_score", formatDouble(result.f1_score)},
            {"inference_time_ms", formatDouble(result.inference_time)}
        });
    }
    std::ofstream file(output_file);
    if (file.is_open()) {
        file << j.dump(4); // Pretty print with 4-space indentation
        file.close();
        std::cout << "JSON report generated: " << output_file << std::endl;
    } else {
        std::cerr << "Error: Unable to open file: " << output_file << std::endl;
    }
}

// Function for model evaluation
std::vector<EvaluationResult> evaluateModels() {
    // Evaluation results for different modalities
    std::vector<EvaluationResult> results = {
        {"Text", 0.92, 0.89, 120.0},
        {"Image", 0.88, 0.87, 200.0},
        {"Audio", 0.85, 0.83, 150.0},
        {"Fusion", 0.93, 0.91, 250.0}
    };
    return results;
}

// Function to print a summary of evaluation results
void printSummary(const std::vector<EvaluationResult>& results) {
    std::cout << "\n====== Evaluation Summary ======\n";
    for (const auto& result : results) {
        std::cout << "Modality: " << result.modality 
                  << " | Accuracy: " << formatDouble(result.accuracy)
                  << " | F1-Score: " << formatDouble(result.f1_score)
                  << " | Inference Time: " << formatDouble(result.inference_time) << " ms\n";
    }
    std::cout << "=================================\n";
}

// Function to measure time taken for report generation
void measurePerformance(const std::function<void()>& func) {
    auto start_time = std::chrono::high_resolution_clock::now();
    func();
    auto end_time = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end_time - start_time);
    std::cout << "Operation completed in " << duration.count() << " ms\n";
}

int main() {
    // Evaluation
    std::vector<EvaluationResult> results = evaluateModels();

    // Generate reports in different formats
    measurePerformance([&results]() {
        generateCSVReport(results, "multimodal_report.csv");
        generateTextReport(results, "multimodal_report.txt");
        generateJSONReport(results, "multimodal_report.json");
    });

    // Print a summary of results
    printSummary(results);

    return 0;
}