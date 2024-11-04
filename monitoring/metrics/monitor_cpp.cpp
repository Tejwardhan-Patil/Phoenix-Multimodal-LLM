#include <iostream>
#include <chrono>
#include <thread>
#include <fstream>
#include <vector>
#include <numeric>
#include <algorithm>
#include <iomanip>
#include <map>

class PerformanceMonitor {
private:
    struct Request {
        double latency;
        std::string type;
        bool success;
    };

    std::vector<Request> requests;
    double start_time;
    double total_requests;
    double total_errors;
    std::ofstream log_file;

public:
    PerformanceMonitor(const std::string& log_path) : total_requests(0), total_errors(0) {
        start_time = std::chrono::duration_cast<std::chrono::milliseconds>(
            std::chrono::system_clock::now().time_since_epoch()
        ).count();
        log_file.open(log_path, std::ios::app);
        if (!log_file.is_open()) {
            throw std::runtime_error("Unable to open log file.");
        }
        log_file << "=== Performance Monitoring Started ===\n";
    }

    ~PerformanceMonitor() {
        if (log_file.is_open()) {
            log_file << "=== Performance Monitoring Ended ===\n";
            log_file.close();
        }
    }

    void log_request(double latency, const std::string& type, bool success) {
        Request req = {latency, type, success};
        requests.push_back(req);
        total_requests++;
        if (!success) {
            total_errors++;
        }

        if (log_file.is_open()) {
            log_file << "Request Type: " << type << ", Latency: " << latency << " ms, Success: "
                     << (success ? "Yes" : "No") << "\n";
        }
    }

    void log_error(const std::string& error_message) {
        total_errors++;
        if (log_file.is_open()) {
            log_file << "Error: " << error_message << "\n";
        }
    }

    double calculate_percentile(double percentile) {
        if (requests.empty()) return 0.0;
        std::vector<double> latencies;
        for (const auto& req : requests) {
            latencies.push_back(req.latency);
        }
        std::sort(latencies.begin(), latencies.end());

        size_t index = static_cast<size_t>(percentile / 100.0 * latencies.size());
        return latencies[index];
    }

    void print_detailed_stats() {
        if (requests.empty()) return;

        std::vector<double> latencies;
        for (const auto& req : requests) {
            latencies.push_back(req.latency);
        }

        double total_latency = std::accumulate(latencies.begin(), latencies.end(), 0.0);
        double avg_latency = total_latency / latencies.size();
        double median_latency = calculate_percentile(50);
        double p95_latency = calculate_percentile(95);
        double p99_latency = calculate_percentile(99);

        double uptime = std::chrono::duration_cast<std::chrono::milliseconds>(
            std::chrono::system_clock::now().time_since_epoch()
        ).count() - start_time;

        std::cout << std::fixed << std::setprecision(2);
        std::cout << "Performance Report\n";
        std::cout << "------------------------------------\n";
        std::cout << "Total Requests: " << total_requests << "\n";
        std::cout << "Total Errors: " << total_errors << "\n";
        std::cout << "Average Latency: " << avg_latency << " ms\n";
        std::cout << "Median Latency (50th Percentile): " << median_latency << " ms\n";
        std::cout << "95th Percentile Latency: " << p95_latency << " ms\n";
        std::cout << "99th Percentile Latency: " << p99_latency << " ms\n";
        std::cout << "Uptime: " << uptime / 1000 << " seconds\n";
        std::cout << "------------------------------------\n";
    }

    void print_request_types_breakdown() {
        if (requests.empty()) return;

        std::map<std::string, int> request_type_count;
        for (const auto& req : requests) {
            request_type_count[req.type]++;
        }

        std::cout << "Request Type Breakdown\n";
        std::cout << "------------------------------------\n";
        for (const auto& entry : request_type_count) {
            std::cout << "Type: " << entry.first << ", Count: " << entry.second << "\n";
        }
        std::cout << "------------------------------------\n";
    }

    void print_latency_histogram() {
        if (requests.empty()) return;

        std::vector<double> latencies;
        for (const auto& req : requests) {
            latencies.push_back(req.latency);
        }

        int bins[10] = {0}; // Histogram with 10 bins

        for (const auto& latency : latencies) {
            int bin = static_cast<int>(latency / 10); // Grouping latency in 10ms intervals
            if (bin < 10) {
                bins[bin]++;
            } else {
                bins[9]++; // All latencies above 100ms fall into the last bin
            }
        }

        std::cout << "Latency Histogram (ms)\n";
        std::cout << "------------------------------------\n";
        for (int i = 0; i < 10; ++i) {
            std::cout << "[" << i * 10 << "-" << (i + 1) * 10 << " ms]: " << bins[i] << "\n";
        }
        std::cout << "------------------------------------\n";
    }
};

int main() {
    PerformanceMonitor monitor("performance_log.txt");

    // Requests and logging
    for (int i = 0; i < 200; ++i) {
        double latency = (rand() % 100) + 20;  // Latencies between 20ms and 120ms
        std::string request_type = (i % 3 == 0) ? "GET" : (i % 3 == 1) ? "POST" : "PUT";  // Different request types
        bool success = (rand() % 10 < 8);  // 80% success rate

        monitor.log_request(latency, request_type, success);

        if (!success) {
            monitor.log_error("Failed to process request: " + request_type);
        }

        std::this_thread::sleep_for(std::chrono::milliseconds(50));  // Delay between requests
    }

    monitor.print_detailed_stats();
    monitor.print_request_types_breakdown();
    monitor.print_latency_histogram();
    
    return 0;
}