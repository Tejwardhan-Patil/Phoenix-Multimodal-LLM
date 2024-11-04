#include <iostream>
#include <fstream>
#include <string>
#include <chrono>
#include <mutex>
#include <thread>
#include <iomanip>
#include <sstream>
#include <vector>
#include <memory>
#include <condition_variable>

// Log output interface for extensibility
class LogOutput {
public:
    virtual ~LogOutput() = default;
    virtual void write(const std::string& message) = 0;
};

class ConsoleOutput : public LogOutput {
public:
    void write(const std::string& message) override {
        std::cout << message << std::endl;
    }
};

class FileOutput : public LogOutput {
public:
    explicit FileOutput(const std::string& filename) : filename_(filename) {
        rotateLogFile();
    }

    void write(const std::string& message) override {
        std::lock_guard<std::mutex> lock(fileMutex_);
        if (!logFile_.is_open()) {
            rotateLogFile();
        }
        logFile_ << message << std::endl;
    }

private:
    void rotateLogFile() {
        if (logFile_.is_open()) {
            logFile_.close();
        }
        logFile_.open(filename_, std::ios::out | std::ios::app);
        if (!logFile_) {
            std::cerr << "Failed to open log file: " << filename_ << std::endl;
        }
    }

    std::ofstream logFile_;
    std::string filename_;
    std::mutex fileMutex_;
};

class NetworkOutput : public LogOutput {
public:
    explicit NetworkOutput(const std::string& server) : server_(server) {
        // Network connection setup
        std::cout << "Connecting to server: " << server << std::endl;
    }

    void write(const std::string& message) override {
        // Network logging
        std::cout << "Sending log to server [" << server_ << "]: " << message << std::endl;
    }

private:
    std::string server_;
};

// Logger class with log rotation and multiple output options
class Logger {
public:
    enum LogLevel {
        INFO,
        WARNING,
        ERROR,
        DEBUG
    };

    static Logger& getInstance() {
        static Logger instance;
        return instance;
    }

    void log(const std::string& message, LogLevel level = INFO) {
        std::lock_guard<std::mutex> lock(mutex_);
        std::string logMessage = formatMessage(message, level);
        for (const auto& output : outputs_) {
            output->write(logMessage);
        }
    }

    void addOutput(std::shared_ptr<LogOutput> output) {
        std::lock_guard<std::mutex> lock(mutex_);
        outputs_.push_back(output);
    }

    void removeAllOutputs() {
        std::lock_guard<std::mutex> lock(mutex_);
        outputs_.clear();
    }

private:
    Logger() = default;
    ~Logger() = default;

    std::string formatMessage(const std::string& message, LogLevel level) {
        std::ostringstream oss;
        oss << "[" << getTimestamp() << "] [" << getLogLevelString(level) << "] " << message;
        return oss.str();
    }

    std::string getTimestamp() {
        auto now = std::chrono::system_clock::now();
        auto in_time_t = std::chrono::system_clock::to_time_t(now);
        auto milliseconds = std::chrono::duration_cast<std::chrono::milliseconds>(
                                now.time_since_epoch()) % 1000;
        std::tm buf;
        localtime_r(&in_time_t, &buf);

        char timestamp[100];
        strftime(timestamp, sizeof(timestamp), "%Y-%m-%d %H:%M:%S", &buf);
        sprintf(timestamp + strlen(timestamp), ".%03d", static_cast<int>(milliseconds.count()));
        return std::string(timestamp);
    }

    std::string getLogLevelString(LogLevel level) {
        switch (level) {
            case INFO: return "INFO";
            case WARNING: return "WARNING";
            case ERROR: return "ERROR";
            case DEBUG: return "DEBUG";
            default: return "UNKNOWN";
        }
    }

    std::vector<std::shared_ptr<LogOutput>> outputs_;
    std::mutex mutex_;
};

// Asynchronous logger to handle high-throughput logging
class AsyncLogger {
public:
    AsyncLogger() : stopLogging_(false) {
        loggingThread_ = std::thread(&AsyncLogger::processLogs, this);
    }

    ~AsyncLogger() {
        {
            std::lock_guard<std::mutex> lock(queueMutex_);
            stopLogging_ = true;
        }
        cv_.notify_all();
        loggingThread_.join();
    }

    void log(const std::string& message, Logger::LogLevel level = Logger::INFO) {
        std::lock_guard<std::mutex> lock(queueMutex_);
        logQueue_.emplace_back(message, level);
        cv_.notify_all();
    }

private:
    void processLogs() {
        while (true) {
            std::unique_lock<std::mutex> lock(queueMutex_);
            cv_.wait(lock, [this] { return !logQueue_.empty() || stopLogging_; });

            while (!logQueue_.empty()) {
                const auto& logEntry = logQueue_.front();
                Logger::getInstance().log(logEntry.first, logEntry.second);
                logQueue_.pop_front();
            }

            if (stopLogging_ && logQueue_.empty()) {
                break;
            }
        }
    }

    std::deque<std::pair<std::string, Logger::LogLevel>> logQueue_;
    std::thread loggingThread_;
    std::mutex queueMutex_;
    std::condition_variable cv_;
    bool stopLogging_;
};

// Usage
int main() {
    Logger& logger = Logger::getInstance();

    // Add console output
    logger.addOutput(std::make_shared<ConsoleOutput>());

    // Add file output
    logger.addOutput(std::make_shared<FileOutput>("logfile.log"));

    // Add network output
    logger.addOutput(std::make_shared<NetworkOutput>("127.0.0.1"));

    // Log some messages
    logger.log("This is an informational message.");
    logger.log("This is a warning message.", Logger::WARNING);
    logger.log("This is an error message.", Logger::ERROR);
    logger.log("Debugging log.", Logger::DEBUG);

    // Using asynchronous logger
    AsyncLogger asyncLogger;
    asyncLogger.log("Asynchronous log message 1", Logger::INFO);
    asyncLogger.log("Asynchronous log message 2", Logger::ERROR);

    std::this_thread::sleep_for(std::chrono::seconds(2));  // Wait for async logs to be processed
    return 0;
}