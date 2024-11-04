#include <iostream>
#include <vector>
#include <random>
#include <cmath>
#include <limits>
#include <chrono>

// Define the hyperparameters structure
struct Hyperparameters {
    double learning_rate;
    int batch_size;
    int num_layers;
    double regularization;
};

// Define a function to evaluate the model performance
double evaluate_model(const Hyperparameters &params) {
    // Evaluation function, returns an error metric based on hyperparameters
    double performance_score = (100.0 / params.learning_rate) * std::log(params.batch_size + params.num_layers + params.regularization);
    performance_score += params.learning_rate * params.regularization; // Add some penalty for regularization
    return performance_score;
}

// Logging function for better output readability
void log_hyperparameters(const Hyperparameters &params, double performance) {
    std::cout << "Evaluating: \n";
    std::cout << "Learning Rate: " << params.learning_rate << "\n";
    std::cout << "Batch Size: " << params.batch_size << "\n";
    std::cout << "Number of Layers: " << params.num_layers << "\n";
    std::cout << "Regularization: " << params.regularization << "\n";
    std::cout << "Performance Score: " << performance << "\n";
    std::cout << "---------------------------------\n";
}

// Function to perform grid search
void grid_search(const std::vector<double>& learning_rates, 
                 const std::vector<int>& batch_sizes, 
                 const std::vector<int>& num_layers,
                 const std::vector<double>& regularizations) {

    double best_performance = std::numeric_limits<double>::max();
    Hyperparameters best_params;

    for (double lr : learning_rates) {
        for (int bs : batch_sizes) {
            for (int layers : num_layers) {
                for (double reg : regularizations) {
                    Hyperparameters params = {lr, bs, layers, reg};
                    double performance = evaluate_model(params);
                    log_hyperparameters(params, performance);

                    if (performance < best_performance) {
                        best_performance = performance;
                        best_params = params;
                    }
                }
            }
        }
    }

    std::cout << "Best Hyperparameters from Grid Search: \n";
    std::cout << "Learning Rate: " << best_params.learning_rate << "\n";
    std::cout << "Batch Size: " << best_params.batch_size << "\n";
    std::cout << "Number of Layers: " << best_params.num_layers << "\n";
    std::cout << "Regularization: " << best_params.regularization << "\n";
    std::cout << "Performance Score: " << best_performance << "\n";
}

// Function to perform random search
void random_search(const std::vector<double>& learning_rates, 
                   const std::vector<int>& batch_sizes, 
                   const std::vector<int>& num_layers, 
                   const std::vector<double>& regularizations,
                   int num_iterations) {

    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_int_distribution<> lr_dist(0, learning_rates.size() - 1);
    std::uniform_int_distribution<> bs_dist(0, batch_sizes.size() - 1);
    std::uniform_int_distribution<> layers_dist(0, num_layers.size() - 1);
    std::uniform_int_distribution<> reg_dist(0, regularizations.size() - 1);

    double best_performance = std::numeric_limits<double>::max();
    Hyperparameters best_params;

    for (int i = 0; i < num_iterations; ++i) {
        Hyperparameters params = {
            learning_rates[lr_dist(gen)],
            batch_sizes[bs_dist(gen)],
            num_layers[layers_dist(gen)],
            regularizations[reg_dist(gen)]
        };

        double performance = evaluate_model(params);
        log_hyperparameters(params, performance);

        if (performance < best_performance) {
            best_performance = performance;
            best_params = params;
        }
    }

    std::cout << "Best Hyperparameters from Random Search: \n";
    std::cout << "Learning Rate: " << best_params.learning_rate << "\n";
    std::cout << "Batch Size: " << best_params.batch_size << "\n";
    std::cout << "Number of Layers: " << best_params.num_layers << "\n";
    std::cout << "Regularization: " << best_params.regularization << "\n";
    std::cout << "Performance Score: " << best_performance << "\n";
}

// Bayesian optimization function
void bayesian_search(const std::vector<double>& learning_rates, 
                     const std::vector<int>& batch_sizes, 
                     const std::vector<int>& num_layers, 
                     const std::vector<double>& regularizations, 
                     int num_iterations) {

    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_int_distribution<> lr_dist(0, learning_rates.size() - 1);
    std::uniform_int_distribution<> bs_dist(0, batch_sizes.size() - 1);
    std::uniform_int_distribution<> layers_dist(0, num_layers.size() - 1);
    std::uniform_int_distribution<> reg_dist(0, regularizations.size() - 1);

    double best_performance = std::numeric_limits<double>::max();
    Hyperparameters best_params;

    // Bayesian-like optimization: start with random samples, then refine
    for (int i = 0; i < num_iterations; ++i) {
        int lr_idx = lr_dist(gen);
        int bs_idx = bs_dist(gen);
        int layers_idx = layers_dist(gen);
        int reg_idx = reg_dist(gen);

        Hyperparameters params = {
            learning_rates[lr_idx],
            batch_sizes[bs_idx],
            num_layers[layers_idx],
            regularizations[reg_idx]
        };

        double performance = evaluate_model(params);
        log_hyperparameters(params, performance);

        // Bayesian refinement
        if (performance < best_performance) {
            best_performance = performance;
            best_params = params;

            // Refine search space around the current best hyperparameters
            lr_idx = std::max(0, lr_idx - 1); // Refining logic
            bs_idx = std::min(static_cast<int>(batch_sizes.size()) - 1, bs_idx + 1);
            layers_idx = std::max(0, layers_idx - 1);
            reg_idx = std::min(static_cast<int>(regularizations.size()) - 1, reg_idx + 1);
        }
    }

    std::cout << "Best Hyperparameters from Bayesian Search: \n";
    std::cout << "Learning Rate: " << best_params.learning_rate << "\n";
    std::cout << "Batch Size: " << best_params.batch_size << "\n";
    std::cout << "Number of Layers: " << best_params.num_layers << "\n";
    std::cout << "Regularization: " << best_params.regularization << "\n";
    std::cout << "Performance Score: " << best_performance << "\n";
}

int main() {
    // Define the search space for hyperparameters
    std::vector<double> learning_rates = {0.01, 0.05, 0.1, 0.5};
    std::vector<int> batch_sizes = {16, 32, 64, 128};
    std::vector<int> num_layers = {1, 2, 3, 4};
    std::vector<double> regularizations = {0.001, 0.01, 0.1};

    // Start timing the grid search
    auto start_grid = std::chrono::high_resolution_clock::now();
    std::cout << "Starting Grid Search..." << std::endl;
    grid_search(learning_rates, batch_sizes, num_layers, regularizations);
    auto end_grid = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> duration_grid = end_grid - start_grid;
    std::cout << "Grid Search Duration: " << duration_grid.count() << " seconds\n\n";

    // Start timing the random search
    auto start_random = std::chrono::high_resolution_clock::now();
    std::cout << "Starting Random Search..." << std::endl;
    random_search(learning_rates, batch_sizes, num_layers, regularizations, 20);
    auto end_random = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> duration_random = end_random - start_random;
    std::cout << "Random Search Duration: " << duration_random.count() << " seconds\n\n";

    // Start timing the Bayesian search
    auto start_bayesian = std::chrono::high_resolution_clock::now();
    std::cout << "Starting Bayesian Search..." << std::endl;
    bayesian_search(learning_rates, batch_sizes, num_layers, regularizations, 20);
    auto end_bayesian = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> duration_bayesian = end_bayesian - start_bayesian;
    std::cout << "Bayesian Search Duration: " << duration_bayesian.count() << " seconds\n\n";

    return 0;
}