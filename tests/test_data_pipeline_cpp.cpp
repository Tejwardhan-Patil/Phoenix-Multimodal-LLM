#include <iostream>
#include <vector>
#include <cassert>
#include "data_loader_cpp.h"
#include "preprocessing_utils_cpp.h"

// Test function for loading data
void test_load_data() {
    DataLoader loader;
    std::vector<std::string> data = loader.load_data("/dataset");

    // Check if data is not empty
    assert(!data.empty() && "Data should not be empty");

    // Check if the size of the dataset is as expected
    assert(data.size() == 100 && "Data size mismatch");

    std::cout << "test_load_data passed!" << std::endl;
}

// Test function for preprocessing text
void test_preprocess_text() {
    PreprocessingUtils preprocessor;

    std::string raw_text = "This is a raw text with some noise.";
    std::string clean_text = preprocessor.clean_text(raw_text);

    // Check that the cleaned text is non-empty
    assert(!clean_text.empty() && "Cleaned text should not be empty");

    // Check the cleaned text content matches expectations
    assert(clean_text == "this is raw text some noise" && "Text preprocessing failed");

    // Additional check for noise removal
    std::string noisy_text = "!!!Noi$y T3xt ###";
    std::string cleaned_noisy_text = preprocessor.clean_text(noisy_text);
    assert(cleaned_noisy_text == "noiy text" && "Noise removal failed");

    std::cout << "test_preprocess_text passed!" << std::endl;
}

// Test function for resizing and normalizing an image
void test_preprocess_image() {
    PreprocessingUtils preprocessor;
    Image raw_image = preprocessor.load_image("/image.jpg");
    Image resized_image = preprocessor.resize_image(raw_image, 256, 256);

    // Check if the image was resized correctly
    assert(resized_image.width == 256 && resized_image.height == 256 && "Image resizing failed");

    // Test normalization of the image
    Image normalized_image = preprocessor.normalize_image(resized_image);
    assert(normalized_image.is_normalized() && "Image normalization failed");

    std::cout << "test_preprocess_image passed!" << std::endl;
}

// Test function for noise reduction and spectrogram computation for audio
void test_audio_preprocessing() {
    PreprocessingUtils preprocessor;
    Audio raw_audio = preprocessor.load_audio("/audio.wav");
    
    // Apply noise reduction
    Audio denoised_audio = preprocessor.reduce_noise(raw_audio);
    assert(denoised_audio.size() > 0 && "Noise reduction failed");

    // Compute spectrogram
    Spectrogram spec = preprocessor.compute_spectrogram(denoised_audio);
    assert(spec.size() > 0 && "Spectrogram computation failed");

    // Check spectrogram dimensions
    assert(spec.width == 128 && spec.height == 128 && "Spectrogram dimensions mismatch");

    std::cout << "test_audio_preprocessing passed!" << std::endl;
}

// Test function for image augmentation
void test_image_augmentation() {
    PreprocessingUtils preprocessor;
    Image raw_image = preprocessor.load_image("/image.jpg");

    // Apply image flipping
    Image flipped_image = preprocessor.flip_image(raw_image, /*horizontal=*/true);
    assert(flipped_image.width == raw_image.width && flipped_image.height == raw_image.height && "Image flipping failed");

    // Apply rotation
    Image rotated_image = preprocessor.rotate_image(flipped_image, 90);
    assert(rotated_image.width == raw_image.height && rotated_image.height == raw_image.width && "Image rotation failed");

    std::cout << "test_image_augmentation passed!" << std::endl;
}

// Test function for text augmentation (synonym replacement, paraphrasing)
void test_text_augmentation() {
    PreprocessingUtils preprocessor;

    std::string original_text = "The quick brown fox jumps over the lazy dog.";
    std::string augmented_text = preprocessor.synonym_replace(original_text);

    assert(augmented_text != original_text && "Text augmentation (synonym replacement) failed");
    std::cout << "test_text_augmentation passed!" << std::endl;
}

// Test for loading large datasets
void test_large_data_loading() {
    DataLoader loader;
    std::vector<std::string> large_data = loader.load_data("/large_dataset");

    // Ensure the large dataset is loaded successfully
    assert(!large_data.empty() && "Large data loading failed");
    assert(large_data.size() > 10000 && "Large dataset size mismatch");

    std::cout << "test_large_data_loading passed!" << std::endl;
}

// Test function for batch processing of data
void test_batch_processing() {
    DataLoader loader;
    PreprocessingUtils preprocessor;

    std::vector<std::string> batch = loader.load_data("/dataset_batch");
    std::vector<std::string> processed_batch;

    for (const auto& item : batch) {
        processed_batch.push_back(preprocessor.clean_text(item));
    }

    assert(processed_batch.size() == batch.size() && "Batch processing size mismatch");
    std::cout << "test_batch_processing passed!" << std::endl;
}

// Edge case test for empty dataset
void test_empty_dataset() {
    DataLoader loader;
    std::vector<std::string> empty_data = loader.load_data("/empty_dataset");

    assert(empty_data.empty() && "Empty dataset test failed");
    std::cout << "test_empty_dataset passed!" << std::endl;
}

// Edge case test for corrupted files
void test_corrupted_data() {
    DataLoader loader;
    try {
        std::vector<std::string> corrupted_data = loader.load_data("/corrupted_dataset");
        assert(false && "Corrupted data test failed - exception not thrown");
    } catch (const std::runtime_error& e) {
        std::cout << "Corrupted data test passed with error: " << e.what() << std::endl;
    }
}

// Function to run all tests
void run_all_tests() {
    test_load_data();
    test_preprocess_text();
    test_preprocess_image();
    test_audio_preprocessing();
    test_image_augmentation();
    test_text_augmentation();
    test_large_data_loading();
    test_batch_processing();
    test_empty_dataset();
    test_corrupted_data();
}

// Main function
int main() {
    run_all_tests();
    std::cout << "All tests passed!" << std::endl;
    return 0;
}