#include <iostream>
#include <vector>
#include <Eigen/Dense> 

using namespace Eigen;

// Function to perform element-wise fusion of feature vectors (concatenate or sum)
VectorXd fuse_features(const VectorXd& text_features, const VectorXd& image_features, const VectorXd& audio_features, const std::string& fusion_method) {
    if (fusion_method == "concat") {
        // Concatenation fusion (concatenation of the vectors)
        VectorXd fused_features(text_features.size() + image_features.size() + audio_features.size());
        fused_features << text_features, image_features, audio_features;
        return fused_features;
    } else if (fusion_method == "sum") {
        // Summing the feature vectors
        if (text_features.size() == image_features.size() && image_features.size() == audio_features.size()) {
            return text_features + image_features + audio_features;
        } else {
            std::cerr << "Feature sizes do not match for sum fusion!" << std::endl;
            exit(1);
        }
    } else {
        std::cerr << "Invalid fusion method!" << std::endl;
        exit(1);
    }
}

// Forward pass for a multimodal model
VectorXd multimodal_forward(const VectorXd& fused_features, const MatrixXd& weights, const VectorXd& biases) {
    // Forward pass through a fully connected layer (weights * features + bias)
    return (weights * fused_features) + biases;
}

// Function to normalize the input feature vectors before fusion
VectorXd normalize_features(const VectorXd& features) {
    // Normalizing the features to have unit length (L2 norm)
    double norm = features.norm();
    if (norm == 0) {
        std::cerr << "Feature vector has zero norm!" << std::endl;
        return features;
    }
    return features / norm;
}

// Function to apply a non-linear activation function (ReLU)
VectorXd apply_activation(const VectorXd& input, const std::string& activation_function) {
    VectorXd activated(input.size());
    
    if (activation_function == "relu") {
        // ReLU activation function (max(0, x))
        for (int i = 0; i < input.size(); ++i) {
            activated[i] = std::max(0.0, input[i]);
        }
    } else if (activation_function == "sigmoid") {
        // Sigmoid activation function (1 / (1 + exp(-x)))
        for (int i = 0; i < input.size(); ++i) {
            activated[i] = 1 / (1 + std::exp(-input[i]));
        }
    } else {
        std::cerr << "Invalid activation function!" << std::endl;
        exit(1);
    }

    return activated;
}

// Main function
int main() {
    // Initialize feature vectors
    VectorXd text_features(300);  // 300-dimensional text features
    VectorXd image_features(512); // 512-dimensional image features
    VectorXd audio_features(128); // 128-dimensional audio features

    // Fill feature vectors with data
    text_features.setRandom();
    image_features.setRandom();
    audio_features.setRandom();

    // Normalize features before fusion
    text_features = normalize_features(text_features);
    image_features = normalize_features(image_features);
    audio_features = normalize_features(audio_features);

    // Fusion method
    std::string fusion_method = "concat"; 

    // Fuse the features
    VectorXd fused_features = fuse_features(text_features, image_features, audio_features, fusion_method);

    // Weights and biases for a simple fully connected layer
    MatrixXd weights(1000, fused_features.size());  // Weight matrix for 1000 output neurons
    weights.setRandom(); 
    VectorXd biases(1000);  // Bias vector
    biases.setRandom(); 

    // Forward pass through the multimodal model
    VectorXd output = multimodal_forward(fused_features, weights, biases);

    // Apply activation function to the output
    output = apply_activation(output, "relu");

    // Output the results
    std::cout << "Multimodal fusion output after activation: " << std::endl;
    std::cout << output.transpose() << std::endl;

    return 0;
}

// Further utility functions to handle larger feature sizes, batch processing, etc
// Function to perform batch normalization (used for large batches of features)
MatrixXd batch_normalization(const MatrixXd& features_batch) {
    MatrixXd normalized_batch(features_batch.rows(), features_batch.cols());

    // Normalize each feature vector in the batch
    for (int i = 0; i < features_batch.rows(); ++i) {
        normalized_batch.row(i) = normalize_features(features_batch.row(i));
    }

    return normalized_batch;
}

// Function to process a batch of fused features through the multimodal model
MatrixXd process_batch(const MatrixXd& fused_features_batch, const MatrixXd& weights, const VectorXd& biases, const std::string& activation_function) {
    MatrixXd batch_output(fused_features_batch.rows(), weights.rows());

    // Perform forward pass and activation for each input in the batch
    for (int i = 0; i < fused_features_batch.rows(); ++i) {
        VectorXd output = multimodal_forward(fused_features_batch.row(i), weights, biases);
        batch_output.row(i) = apply_activation(output, activation_function);
    }

    return batch_output;
}

// Function to handle multiple batches of inputs
void process_batches() {
    int batch_size = 32;  // Batch size
    int num_batches = 10;  // Number of batches to process

    // Initialize random batches of features
    MatrixXd text_batch(batch_size, 300);   // Batch of text features
    MatrixXd image_batch(batch_size, 512);  // Batch of image features
    MatrixXd audio_batch(batch_size, 128);  // Batch of audio features

    text_batch.setRandom();
    image_batch.setRandom();
    audio_batch.setRandom();

    // Initialize weights and biases for the model
    MatrixXd weights(1000, text_batch.cols() + image_batch.cols() + audio_batch.cols());
    weights.setRandom();
    VectorXd biases(1000);
    biases.setRandom();

    for (int i = 0; i < num_batches; ++i) {
        // Normalize and fuse the features for each batch
        MatrixXd fused_batch(batch_size, text_batch.cols() + image_batch.cols() + audio_batch.cols());

        for (int j = 0; j < batch_size; ++j) {
            fused_batch.row(j) = fuse_features(text_batch.row(j), image_batch.row(j), audio_batch.row(j), "concat");
        }

        // Process the fused batch through the model
        MatrixXd batch_output = process_batch(fused_batch, weights, biases, "relu");

        // Output the batch results
        std::cout << "Batch " << i + 1 << " output:" << std::endl;
        std::cout << batch_output << std::endl;
    }
}

// Function to compute the softmax of a vector
VectorXd softmax(const VectorXd& input) {
    VectorXd exps = input.array().exp();
    return exps / exps.sum();
}

// Function to handle multiple types of fusion for a batch of inputs
MatrixXd fuse_batch(const MatrixXd& text_batch, const MatrixXd& image_batch, const MatrixXd& audio_batch, const std::string& fusion_method) {
    int batch_size = text_batch.rows();
    MatrixXd fused_batch(batch_size, text_batch.cols() + image_batch.cols() + audio_batch.cols());

    for (int i = 0; i < batch_size; ++i) {
        fused_batch.row(i) = fuse_features(text_batch.row(i), image_batch.row(i), audio_batch.row(i), fusion_method);
    }

    return fused_batch;
}

// Function to apply softmax across a batch of outputs
MatrixXd apply_softmax_batch(const MatrixXd& input_batch) {
    MatrixXd softmax_output(input_batch.rows(), input_batch.cols());

    for (int i = 0; i < input_batch.rows(); ++i) {
        softmax_output.row(i) = softmax(input_batch.row(i));
    }

    return softmax_output;
}

// Function to handle learning rate-based updates to the model weights (gradient descent)
void update_weights(MatrixXd& weights, const MatrixXd& gradients, double learning_rate) {
    weights -= learning_rate * gradients;
}

// Function to compute the gradients for the weights given the batch of inputs and corresponding outputs
MatrixXd compute_gradients(const MatrixXd& fused_batch, const MatrixXd& outputs, const MatrixXd& targets) {
    MatrixXd errors = outputs - targets;  // Simple error calculation (can be changed based on loss function)
    MatrixXd gradients = errors.transpose() * fused_batch;
    return gradients;
}

// Training loop function for the multimodal model
void train_multimodal_model(MatrixXd& weights, VectorXd& biases, const std::vector<MatrixXd>& text_batches, const std::vector<MatrixXd>& image_batches, const std::vector<MatrixXd>& audio_batches, const std::vector<MatrixXd>& target_batches, int num_epochs, double learning_rate) {
    int batch_size = text_batches[0].rows();

    for (int epoch = 0; epoch < num_epochs; ++epoch) {
        double epoch_loss = 0.0;

        for (int batch_idx = 0; batch_idx < text_batches.size(); ++batch_idx) {
            // Normalize and fuse the input batches
            MatrixXd fused_batch = fuse_batch(text_batches[batch_idx], image_batches[batch_idx], audio_batches[batch_idx], "concat");

            // Forward pass through the model
            MatrixXd batch_output = process_batch(fused_batch, weights, biases, "relu");

            // Apply softmax to the output
            MatrixXd softmax_output = apply_softmax_batch(batch_output);

            // Compute gradients
            MatrixXd gradients = compute_gradients(fused_batch, softmax_output, target_batches[batch_idx]);

            // Update weights using gradient descent
            update_weights(weights, gradients, learning_rate);

            // Compute batch loss (simple cross-entropy)
            for (int i = 0; i < batch_size; ++i) {
                epoch_loss += -std::log(softmax_output.row(i).dot(target_batches[batch_idx].row(i)));
            }
        }

        // Average the loss
        epoch_loss /= text_batches.size() * batch_size;

        std::cout << "Epoch " << epoch + 1 << " completed. Loss: " << epoch_loss << std::endl;
    }
}

// Function to initialize batches of random target data (for supervised learning)
std::vector<MatrixXd> initialize_random_targets(int num_batches, int batch_size, int num_classes) {
    std::vector<MatrixXd> target_batches;
    
    for (int i = 0; i < num_batches; ++i) {
        MatrixXd targets(batch_size, num_classes);
        targets.setRandom();
        targets = (targets.array() > 0).cast<double>();  // Random binary target generation for classification
        target_batches.push_back(targets);
    }

    return target_batches;
}

// Function to generate random input batches for testing (for text, image, and audio)
std::vector<MatrixXd> generate_random_batches(int num_batches, int batch_size, int feature_size) {
    std::vector<MatrixXd> batches;

    for (int i = 0; i < num_batches; ++i) {
        MatrixXd batch(batch_size, feature_size);
        batch.setRandom();
        batches.push_back(batch);
    }

    return batches;
}

// Main function to run the training of the multimodal model
int main() {
    int num_batches = 10;      // Number of batches
    int batch_size = 32;       // Size of each batch
    int text_feature_size = 300;  // Dimensionality of text features
    int image_feature_size = 512; // Dimensionality of image features
    int audio_feature_size = 128; // Dimensionality of audio features
    int num_classes = 10;      // Number of output classes (for classification)

    // Generate random input batches
    std::vector<MatrixXd> text_batches = generate_random_batches(num_batches, batch_size, text_feature_size);
    std::vector<MatrixXd> image_batches = generate_random_batches(num_batches, batch_size, image_feature_size);
    std::vector<MatrixXd> audio_batches = generate_random_batches(num_batches, batch_size, audio_feature_size);

    // Generate random target batches (binary classification targets)
    std::vector<MatrixXd> target_batches = initialize_random_targets(num_batches, batch_size, num_classes);

    // Initialize weights and biases for the multimodal model
    MatrixXd weights(num_classes, text_feature_size + image_feature_size + audio_feature_size);
    weights.setRandom();
    VectorXd biases(num_classes);
    biases.setRandom();

    // Set training parameters
    int num_epochs = 20;
    double learning_rate = 0.01;

    // Train the model
    train_multimodal_model(weights, biases, text_batches, image_batches, audio_batches, target_batches, num_epochs, learning_rate);

    return 0;
}

// Function to compute accuracy of the model
double compute_accuracy(const MatrixXd& predictions, const MatrixXd& targets) {
    int correct_predictions = 0;

    for (int i = 0; i < predictions.rows(); ++i) {
        int predicted_class = predictions.row(i).maxCoeffIndex();
        int target_class = targets.row(i).maxCoeffIndex();

        if (predicted_class == target_class) {
            ++correct_predictions;
        }
    }

    return static_cast<double>(correct_predictions) / predictions.rows();
}

// Function to evaluate the model on test batches
void evaluate_multimodal_model(const MatrixXd& weights, const VectorXd& biases, const std::vector<MatrixXd>& text_batches, const std::vector<MatrixXd>& image_batches, const std::vector<MatrixXd>& audio_batches, const std::vector<MatrixXd>& target_batches) {
    int batch_size = text_batches[0].rows();
    double total_accuracy = 0.0;

    for (int batch_idx = 0; batch_idx < text_batches.size(); ++batch_idx) {
        // Normalize and fuse the input batches
        MatrixXd fused_batch = fuse_batch(text_batches[batch_idx], image_batches[batch_idx], audio_batches[batch_idx], "concat");

        // Forward pass through the model
        MatrixXd batch_output = process_batch(fused_batch, weights, biases, "relu");

        // Apply softmax to the output
        MatrixXd softmax_output = apply_softmax_batch(batch_output);

        // Compute accuracy for the batch
        double batch_accuracy = compute_accuracy(softmax_output, target_batches[batch_idx]);
        total_accuracy += batch_accuracy;

        std::cout << "Batch " << batch_idx + 1 << " accuracy: " << batch_accuracy * 100 << "%" << std::endl;
    }

    // Average accuracy over all batches
    total_accuracy /= text_batches.size();
    std::cout << "Overall accuracy: " << total_accuracy * 100 << "%" << std::endl;
}

// Function to apply dropout to the feature vectors during training
MatrixXd apply_dropout(const MatrixXd& input_batch, double dropout_rate) {
    MatrixXd output_batch(input_batch.rows(), input_batch.cols());
    for (int i = 0; i < input_batch.rows(); ++i) {
        for (int j = 0; j < input_batch.cols(); ++j) {
            if ((rand() / (double)RAND_MAX) > dropout_rate) {
                output_batch(i, j) = input_batch(i, j);  // Keep the unit
            } else {
                output_batch(i, j) = 0;  // Drop the unit
            }
        }
    }
    return output_batch;
}

// Function to apply L2 regularization to weights during training
double l2_regularization(const MatrixXd& weights, double lambda) {
    return lambda * weights.squaredNorm();
}

// Modified training loop with dropout and L2 regularization
void train_multimodal_model_with_regularization(MatrixXd& weights, VectorXd& biases, const std::vector<MatrixXd>& text_batches, const std::vector<MatrixXd>& image_batches, const std::vector<MatrixXd>& audio_batches, const std::vector<MatrixXd>& target_batches, int num_epochs, double learning_rate, double dropout_rate, double lambda) {
    int batch_size = text_batches[0].rows();

    for (int epoch = 0; epoch < num_epochs; ++epoch) {
        double epoch_loss = 0.0;

        for (int batch_idx = 0; batch_idx < text_batches.size(); ++batch_idx) {
            // Normalize and fuse the input batches
            MatrixXd fused_batch = fuse_batch(text_batches[batch_idx], image_batches[batch_idx], audio_batches[batch_idx], "concat");

            // Apply dropout to the fused features
            fused_batch = apply_dropout(fused_batch, dropout_rate);

            // Forward pass through the model
            MatrixXd batch_output = process_batch(fused_batch, weights, biases, "relu");

            // Apply softmax to the output
            MatrixXd softmax_output = apply_softmax_batch(batch_output);

            // Compute gradients
            MatrixXd gradients = compute_gradients(fused_batch, softmax_output, target_batches[batch_idx]);

            // Update weights using gradient descent with L2 regularization
            update_weights(weights, gradients, learning_rate);
            epoch_loss += -std::log(softmax_output.row(0).dot(target_batches[batch_idx].row(0)));

            // Apply L2 regularization
            epoch_loss += l2_regularization(weights, lambda);
        }

        // Average the loss
        epoch_loss /= text_batches.size() * batch_size;

        std::cout << "Epoch " << epoch + 1 << " completed. Loss: " << epoch_loss << std::endl;
    }
}

// Function to visualize the fused features for better interpretability
void visualize_fused_features(const VectorXd& fused_features) {
    std::cout << "Visualizing fused features..." << std::endl;
    for (int i = 0; i < fused_features.size(); ++i) {
        std::cout << "Feature " << i + 1 << ": " << fused_features[i] << std::endl;
    }
}

// Function to visualize batch outputs
void visualize_batch_output(const MatrixXd& batch_output) {
    std::cout << "Visualizing batch output..." << std::endl;
    for (int i = 0; i < batch_output.rows(); ++i) {
        std::cout << "Output " << i + 1 << ": " << batch_output.row(i) << std::endl;
    }
}

// Modified evaluation function with more detailed logging and visualization
void evaluate_model_with_visualization(const MatrixXd& weights, const VectorXd& biases, const std::vector<MatrixXd>& text_batches, const std::vector<MatrixXd>& image_batches, const std::vector<MatrixXd>& audio_batches, const std::vector<MatrixXd>& target_batches) {
    int batch_size = text_batches[0].rows();
    double total_accuracy = 0.0;

    for (int batch_idx = 0; batch_idx < text_batches.size(); ++batch_idx) {
        // Normalize and fuse the input batches
        MatrixXd fused_batch = fuse_batch(text_batches[batch_idx], image_batches[batch_idx], audio_batches[batch_idx], "concat");

        // Visualize fused features
        visualize_fused_features(fused_batch.row(0));

        // Forward pass through the model
        MatrixXd batch_output = process_batch(fused_batch, weights, biases, "relu");

        // Visualize batch outputs
        visualize_batch_output(batch_output);

        // Apply softmax to the output
        MatrixXd softmax_output = apply_softmax_batch(batch_output);

        // Compute accuracy for the batch
        double batch_accuracy = compute_accuracy(softmax_output, target_batches[batch_idx]);
        total_accuracy += batch_accuracy;

        std::cout << "Batch " << batch_idx + 1 << " accuracy: " << batch_accuracy * 100 << "%" << std::endl;
    }

    // Average accuracy over all batches
    total_accuracy /= text_batches.size();
    std::cout << "Overall accuracy: " << total_accuracy * 100 << "%" << std::endl;
}

// Function to save model weights to a file for future use
void save_model(const MatrixXd& weights, const VectorXd& biases, const std::string& filename) {
    std::ofstream file;
    file.open(filename);
    
    if (!file.is_open()) {
        std::cerr << "Error opening file for writing." << std::endl;
        return;
    }

    // Save weights
    for (int i = 0; i < weights.rows(); ++i) {
        for (int j = 0; j < weights.cols(); ++j) {
            file << weights(i, j) << " ";
        }
        file << std::endl;
    }

    // Save biases
    for (int i = 0; i < biases.size(); ++i) {
        file << biases[i] << " ";
    }
    
    file.close();
    std::cout << "Model saved to " << filename << std::endl;
}

// Function to load model weights from a file
void load_model(MatrixXd& weights, VectorXd& biases, const std::string& filename) {
    std::ifstream file;
    file.open(filename);
    
    if (!file.is_open()) {
        std::cerr << "Error opening file for reading." << std::endl;
        return;
    }

    // Load weights
    for (int i = 0; i < weights.rows(); ++i) {
        for (int j = 0; j < weights.cols(); ++j) {
            file >> weights(i, j);
        }
    }

    // Load biases
    for (int i = 0; i < biases.size(); ++i) {
        file >> biases[i];
    }
    
    file.close();
    std::cout << "Model loaded from " << filename << std::endl;
}

// Main function for testing model saving and loading
int main() {
    int num_batches = 10;        // Number of batches
    int batch_size = 32;         // Size of each batch
    int text_feature_size = 300; // Dimensionality of text features
    int image_feature_size = 512; // Dimensionality of image features
    int audio_feature_size = 128; // Dimensionality of audio features
    int num_classes = 10;        // Number of output classes (for classification)

    // Generate random input batches
    std::vector<MatrixXd> text_batches = generate_random_batches(num_batches, batch_size, text_feature_size);
    std::vector<MatrixXd> image_batches = generate_random_batches(num_batches, batch_size, image_feature_size);
    std::vector<MatrixXd> audio_batches = generate_random_batches(num_batches, batch_size, audio_feature_size);

    // Generate random target batches (binary classification targets)
    std::vector<MatrixXd> target_batches = initialize_random_targets(num_batches, batch_size, num_classes);

    // Initialize weights and biases for the multimodal model
    MatrixXd weights(num_classes, text_feature_size + image_feature_size + audio_feature_size);
    weights.setRandom();
    VectorXd biases(num_classes);
    biases.setRandom();

    // Set training parameters
    int num_epochs = 20;
    double learning_rate = 0.01;
    double dropout_rate = 0.5;
    double lambda = 0.01;

    // Train the model with regularization
    train_multimodal_model_with_regularization(weights, biases, text_batches, image_batches, audio_batches, target_batches, num_epochs, learning_rate, dropout_rate, lambda);

    // Save the trained model
    save_model(weights, biases, "multimodal_model.txt");

    // Load the model back
    load_model(weights, biases, "multimodal_model.txt");

    // Evaluate the model
    evaluate_model_with_visualization(weights, biases, text_batches, image_batches, audio_batches, target_batches);

    return 0;
}