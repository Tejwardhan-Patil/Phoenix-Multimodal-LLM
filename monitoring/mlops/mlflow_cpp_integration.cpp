#include <mlflow/mlflow.h>
#include <string>
#include <iostream>
#include <fstream>
#include <cstdlib>
#include <vector>

// Utility function to check if environment variable is set
std::string get_env_var(const std::string& var) {
    const char* value = std::getenv(var.c_str());
    if (value == nullptr) {
        std::cerr << "Environment variable " << var << " is not set." << std::endl;
        std::exit(EXIT_FAILURE);
    }
    return std::string(value);
}

// Initialize MLflow tracking URI and experiment details
void init_mlflow(const std::string& experiment_name) {
    std::string tracking_uri = get_env_var("MLFLOW_TRACKING_URI");
    mlflow::MlflowClient client(tracking_uri);

    try {
        mlflow::Experiment experiment = client.get_experiment_by_name(experiment_name);
        if (experiment.experiment_id.empty()) {
            client.create_experiment(experiment_name);
            std::cout << "Created new experiment: " << experiment_name << std::endl;
        } else {
            std::cout << "Using existing experiment: " << experiment_name << std::endl;
        }
    } catch (const std::exception& e) {
        std::cerr << "Error initializing MLflow: " << e.what() << std::endl;
        std::exit(EXIT_FAILURE);
    }
}

// Start a new MLflow run and log basic information
std::string start_mlflow_run(const std::string& experiment_name, const std::string& run_name) {
    mlflow::MlflowClient client;

    std::string run_id;
    try {
        run_id = client.create_run(experiment_name, run_name);
        std::cout << "Started MLflow run: " << run_id << std::endl;
    } catch (const std::exception& e) {
        std::cerr << "Error starting MLflow run: " << e.what() << std::endl;
        std::exit(EXIT_FAILURE);
    }

    return run_id;
}

// Log a list of parameters to the current MLflow run
void log_mlflow_params(const std::string& run_id, const std::vector<std::pair<std::string, std::string>>& params) {
    mlflow::MlflowClient client;
    for (const auto& param : params) {
        try {
            client.log_param(run_id, param.first, param.second);
            std::cout << "Logged parameter: " << param.first << " = " << param.second << std::endl;
        } catch (const std::exception& e) {
            std::cerr << "Error logging parameter: " << param.first << " - " << e.what() << std::endl;
        }
    }
}

// Log a list of metrics to the current MLflow run
void log_mlflow_metrics(const std::string& run_id, const std::vector<std::pair<std::string, double>>& metrics) {
    mlflow::MlflowClient client;
    for (const auto& metric : metrics) {
        try {
            client.log_metric(run_id, metric.first, metric.second);
            std::cout << "Logged metric: " << metric.first << " = " << metric.second << std::endl;
        } catch (const std::exception& e) {
            std::cerr << "Error logging metric: " << metric.first << " - " << e.what() << std::endl;
        }
    }
}

// Log model information to MLflow
void log_mlflow_model(const std::string& run_id, const std::string& model_path, const std::string& model_name) {
    mlflow::MlflowClient client;
    try {
        client.log_artifact(run_id, model_path, "models/" + model_name);
        std::cout << "Logged model: " << model_name << " from path " << model_path << std::endl;
    } catch (const std::exception& e) {
        std::cerr << "Error logging model: " << e.what() << std::endl;
    }
}

// Log artifacts (like configuration files) to MLflow
void log_mlflow_artifact(const std::string& run_id, const std::string& file_path, const std::string& artifact_name) {
    mlflow::MlflowClient client;
    try {
        client.log_artifact(run_id, file_path, "artifacts/" + artifact_name);
        std::cout << "Logged artifact: " << artifact_name << " from path " << file_path << std::endl;
    } catch (const std::exception& e) {
        std::cerr << "Error logging artifact: " << e.what() << std::endl;
    }
}

// Save configuration to a file and log it to MLflow
void save_and_log_config(const std::string& run_id, const std::string& config_path) {
    std::ofstream config_file(config_path);
    if (config_file.is_open()) {
        config_file << "learning_rate=0.001\n";
        config_file << "batch_size=32\n";
        config_file << "epochs=50\n";
        config_file.close();
        log_mlflow_artifact(run_id, config_path, "training_config.txt");
    } else {
        std::cerr << "Unable to open file for saving configuration." << std::endl;
    }
}

// Log system environment details as parameters
void log_system_info(const std::string& run_id) {
    std::string os_name = get_env_var("OS_NAME");
    std::string cpu_cores = get_env_var("CPU_CORES");
    std::string memory = get_env_var("TOTAL_MEMORY");

    std::vector<std::pair<std::string, std::string>> system_info = {
        {"os_name", os_name},
        {"cpu_cores", cpu_cores},
        {"total_memory", memory}
    };

    log_mlflow_params(run_id, system_info);
}

// Log experiment summary metrics
void log_experiment_summary(const std::string& run_id) {
    std::vector<std::pair<std::string, double>> summary_metrics = {
        {"final_accuracy", 0.92},
        {"final_loss", 0.22},
        {"training_time_seconds", 3600}
    };

    log_mlflow_metrics(run_id, summary_metrics);
}

// End the current MLflow run
void end_mlflow_run(const std::string& run_id) {
    mlflow::MlflowClient client;

    try {
        client.end_run(run_id);
        std::cout << "MLflow run ended: " << run_id << std::endl;
    } catch (const std::exception& e) {
        std::cerr << "Error ending MLflow run: " << e.what() << std::endl;
    }
}

// Model training process
void train_model(const std::string& run_id) {
    // Training process
    std::cout << "Training model..." << std::endl;

    // Log interim metrics during training
    for (int epoch = 1; epoch <= 10; ++epoch) {
        double accuracy = 0.80 + epoch * 0.01;
        double loss = 0.50 - epoch * 0.02;

        std::vector<std::pair<std::string, double>> epoch_metrics = {
            {"epoch_accuracy", accuracy},
            {"epoch_loss", loss}
        };

        log_mlflow_metrics(run_id, epoch_metrics);
        std::cout << "Epoch " << epoch << " completed. Accuracy: " << accuracy << ", Loss: " << loss << std::endl;
    }

    // Save and log the model after training
    std::string model_path = "/model";
    log_mlflow_model(run_id, model_path, "final_model");
}

int main() {
    const std::string experiment_name = "cpp_mlflow_extended_experiment";
    const std::string run_name = "cpp_mlflow_extended_run";

    // Initialize MLflow
    init_mlflow(experiment_name);

    // Start a new MLflow run
    std::string run_id = start_mlflow_run(experiment_name, run_name);

    // Log system information
    log_system_info(run_id);

    // Log initial parameters
    std::vector<std::pair<std::string, std::string>> initial_params = {
        {"learning_rate", "0.001"},
        {"batch_size", "32"},
        {"epochs", "50"}
    };
    log_mlflow_params(run_id, initial_params);

    // Train the model and log metrics
    train_model(run_id);

    // Log final experiment summary
    log_experiment_summary(run_id);

    // Save and log the training configuration
    std::string config_path = "/config";
    save_and_log_config(run_id, config_path);

    // End the run
    end_mlflow_run(run_id);

    return 0;
}