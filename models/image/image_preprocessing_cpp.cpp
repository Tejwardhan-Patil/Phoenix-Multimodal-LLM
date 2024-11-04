#include <opencv2/opencv.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <iostream>
#include <vector>
#include <string>
#include <algorithm>

// Function to resize image
cv::Mat resize_image(const cv::Mat& input_image, int target_width, int target_height) {
    cv::Mat resized_image;
    cv::resize(input_image, resized_image, cv::Size(target_width, target_height));
    return resized_image;
}

// Function to apply Gaussian blur to an image
cv::Mat apply_gaussian_blur(const cv::Mat& input_image, int kernel_size) {
    cv::Mat blurred_image;
    cv::GaussianBlur(input_image, blurred_image, cv::Size(kernel_size, kernel_size), 0);
    return blurred_image;
}

// Function to normalize the image pixels to [0,1]
cv::Mat normalize_image(const cv::Mat& input_image) {
    cv::Mat normalized_image;
    input_image.convertTo(normalized_image, CV_32F, 1.0 / 255.0);
    return normalized_image;
}

// Function to apply a sharpening filter
cv::Mat sharpen_image(const cv::Mat& input_image) {
    cv::Mat sharpened_image;
    cv::Mat kernel = (cv::Mat_<float>(3, 3) <<  0, -1,  0,
                                               -1,  5, -1,
                                                0, -1,  0);
    cv::filter2D(input_image, sharpened_image, input_image.depth(), kernel);
    return sharpened_image;
}

// Function to convert the image to grayscale
cv::Mat convert_to_grayscale(const cv::Mat& input_image) {
    cv::Mat grayscale_image;
    cv::cvtColor(input_image, grayscale_image, cv::COLOR_BGR2GRAY);
    return grayscale_image;
}

// Function to apply edge detection using the Canny algorithm
cv::Mat detect_edges(const cv::Mat& input_image, double lower_threshold, double upper_threshold) {
    cv::Mat edges;
    cv::Canny(input_image, edges, lower_threshold, upper_threshold);
    return edges;
}

// Function to rotate an image by a specific angle
cv::Mat rotate_image(const cv::Mat& input_image, double angle) {
    cv::Mat rotated_image;
    cv::Point2f center(input_image.cols / 2.0F, input_image.rows / 2.0F);
    cv::Mat rotation_matrix = cv::getRotationMatrix2D(center, angle, 1.0);
    cv::warpAffine(input_image, rotated_image, rotation_matrix, input_image.size());
    return rotated_image;
}

// Function to flip an image horizontally or vertically
cv::Mat flip_image(const cv::Mat& input_image, int flip_code) {
    cv::Mat flipped_image;
    cv::flip(input_image, flipped_image, flip_code);
    return flipped_image;
}

// Function to adjust the brightness of an image
cv::Mat adjust_brightness(const cv::Mat& input_image, double alpha, int beta) {
    cv::Mat brightened_image;
    input_image.convertTo(brightened_image, -1, alpha, beta);
    return brightened_image;
}

// Function to adjust the contrast of an image
cv::Mat adjust_contrast(const cv::Mat& input_image, double alpha) {
    cv::Mat contrasted_image;
    input_image.convertTo(contrasted_image, -1, alpha, 0);
    return contrasted_image;
}

// Function to overlay one image on top of another
cv::Mat overlay_images(const cv::Mat& base_image, const cv::Mat& overlay_image, double alpha) {
    cv::Mat blended_image;
    cv::addWeighted(base_image, alpha, overlay_image, 1.0 - alpha, 0.0, blended_image);
    return blended_image;
}

// Function to change image color space to HSV
cv::Mat convert_to_hsv(const cv::Mat& input_image) {
    cv::Mat hsv_image;
    cv::cvtColor(input_image, hsv_image, cv::COLOR_BGR2HSV);
    return hsv_image;
}

// Function to change image color space to LAB
cv::Mat convert_to_lab(const cv::Mat& input_image) {
    cv::Mat lab_image;
    cv::cvtColor(input_image, lab_image, cv::COLOR_BGR2Lab);
    return lab_image;
}

// Function to resize the image and maintain aspect ratio
cv::Mat resize_with_aspect_ratio(const cv::Mat& input_image, int max_width, int max_height) {
    int original_width = input_image.cols;
    int original_height = input_image.rows;
    double aspect_ratio = (double)original_width / original_height;
    int new_width, new_height;

    if (aspect_ratio > 1.0) {
        new_width = max_width;
        new_height = static_cast<int>(max_width / aspect_ratio);
    } else {
        new_width = static_cast<int>(max_height * aspect_ratio);
        new_height = max_height;
    }

    cv::Mat resized_image;
    cv::resize(input_image, resized_image, cv::Size(new_width, new_height));
    return resized_image;
}

// Function to convert an image to binary (black and white) using a threshold
cv::Mat convert_to_binary(const cv::Mat& input_image, double threshold_value) {
    cv::Mat grayscale_image = convert_to_grayscale(input_image);
    cv::Mat binary_image;
    cv::threshold(grayscale_image, binary_image, threshold_value, 255, cv::THRESH_BINARY);
    return binary_image;
}

// Function to detect contours in an image
std::vector<std::vector<cv::Point>> detect_contours(const cv::Mat& input_image) {
    cv::Mat edges = detect_edges(input_image, 100, 200);
    std::vector<std::vector<cv::Point>> contours;
    cv::findContours(edges, contours, cv::RETR_TREE, cv::CHAIN_APPROX_SIMPLE);
    return contours;
}

// Function to draw contours on an image
cv::Mat draw_contours(const cv::Mat& input_image, const std::vector<std::vector<cv::Point>>& contours) {
    cv::Mat contour_image = input_image.clone();
    cv::drawContours(contour_image, contours, -1, cv::Scalar(0, 255, 0), 2);
    return contour_image;
}

// Function to calculate the histogram of an image
cv::Mat calculate_histogram(const cv::Mat& input_image) {
    std::vector<cv::Mat> bgr_planes;
    cv::split(input_image, bgr_planes);

    int hist_size = 256;
    float range[] = { 0, 256 };
    const float* hist_range = { range };
    bool uniform = true;
    bool accumulate = false;

    cv::Mat b_hist, g_hist, r_hist;
    cv::calcHist(&bgr_planes[0], 1, 0, cv::Mat(), b_hist, 1, &hist_size, &hist_range, uniform, accumulate);
    cv::calcHist(&bgr_planes[1], 1, 0, cv::Mat(), g_hist, 1, &hist_size, &hist_range, uniform, accumulate);
    cv::calcHist(&bgr_planes[2], 1, 0, cv::Mat(), r_hist, 1, &hist_size, &hist_range, uniform, accumulate);

    int hist_w = 512, hist_h = 400;
    int bin_w = cv::round((double)hist_w / hist_size);

    cv::Mat hist_image(hist_h, hist_w, CV_8UC3, cv::Scalar(0, 0, 0));

    cv::normalize(b_hist, b_hist, 0, hist_image.rows, cv::NORM_MINMAX, -1, cv::Mat());
    cv::normalize(g_hist, g_hist, 0, hist_image.rows, cv::NORM_MINMAX, -1, cv::Mat());
    cv::normalize(r_hist, r_hist, 0, hist_image.rows, cv::NORM_MINMAX, -1, cv::Mat());

    for (int i = 1; i < hist_size; i++) {
        cv::line(hist_image, cv::Point(bin_w*(i - 1), hist_h - cv::round(b_hist.at<float>(i - 1))),
                 cv::Point(bin_w*(i), hist_h - cv::round(b_hist.at<float>(i))),
                 cv::Scalar(255, 0, 0), 2, 8, 0);
        cv::line(hist_image, cv::Point(bin_w*(i - 1), hist_h - cv::round(g_hist.at<float>(i - 1))),
                 cv::Point(bin_w*(i), hist_h - cv::round(g_hist.at<float>(i))),
                 cv::Scalar(0, 255, 0), 2, 8, 0);
        cv::line(hist_image, cv::Point(bin_w*(i - 1), hist_h - cv::round(r_hist.at<float>(i - 1))),
                 cv::Point(bin_w*(i), hist_h - cv::round(r_hist.at<float>(i))),
                 cv::Scalar(0, 0, 255), 2, 8, 0);
    }

    return hist_image;
}

int main(int argc, char** argv) {
    if (argc < 2) {
        std::cerr << "Usage: " << argv[0] << " <image_path>" << std::endl;
        return -1;
    }

    // Load the image from the provided path
    cv::Mat input_image = cv::imread(argv[1], cv::IMREAD_COLOR);
    if (input_image.empty()) {
        std::cerr << "Error: Could not load image " << argv[1] << std::endl;
        return -1;
    }

    // Preprocessing parameters
    int target_width = 224;
    int target_height = 224;
    int blur_kernel_size = 5;

    // Resize the image
    cv::Mat resized_image = resize_image(input_image, target_width, target_height);

    // Apply Gaussian blur
    cv::Mat blurred_image = apply_gaussian_blur(resized_image, blur_kernel_size);

    // Normalize the image
    cv::Mat normalized_image = normalize_image(blurred_image);

    // Display the processed image
    cv::imshow("Processed Image", normalized_image);
    cv::waitKey(0);

    return 0;
}

// Function to equalize the histogram of a grayscale image
cv::Mat equalize_histogram(const cv::Mat& input_image) {
    cv::Mat grayscale_image = convert_to_grayscale(input_image);
    cv::Mat equalized_image;
    cv::equalizeHist(grayscale_image, equalized_image);
    return equalized_image;
}

// Function to apply adaptive thresholding
cv::Mat adaptive_threshold(const cv::Mat& input_image, double max_value, int adaptive_method, int threshold_type, int block_size, double C) {
    cv::Mat grayscale_image = convert_to_grayscale(input_image);
    cv::Mat thresholded_image;
    cv::adaptiveThreshold(grayscale_image, thresholded_image, max_value, adaptive_method, threshold_type, block_size, C);
    return thresholded_image;
}

// Function to apply bilateral filtering
cv::Mat bilateral_filter(const cv::Mat& input_image, int diameter, double sigma_color, double sigma_space) {
    cv::Mat filtered_image;
    cv::bilateralFilter(input_image, filtered_image, diameter, sigma_color, sigma_space);
    return filtered_image;
}

// Function to apply a median filter to reduce noise
cv::Mat median_filter(const cv::Mat& input_image, int kernel_size) {
    cv::Mat filtered_image;
    cv::medianBlur(input_image, filtered_image, kernel_size);
    return filtered_image;
}

// Function to convert an image to a sepia tone
cv::Mat convert_to_sepia(const cv::Mat& input_image) {
    cv::Mat sepia_image = input_image.clone();
    cv::Mat kernel = (cv::Mat_<float>(3, 3) <<  0.272, 0.534, 0.131,
                                                0.349, 0.686, 0.168,
                                                0.393, 0.769, 0.189);
    cv::transform(input_image, sepia_image, kernel);
    return sepia_image;
}

// Function to perform gamma correction on an image
cv::Mat gamma_correction(const cv::Mat& input_image, double gamma) {
    cv::Mat gamma_corrected_image;
    cv::Mat lut_matrix(1, 256, CV_8UC1);
    uchar* ptr = lut_matrix.ptr();
    for (int i = 0; i < 256; ++i) {
        ptr[i] = cv::saturate_cast<uchar>(pow(i / 255.0, gamma) * 255.0);
    }
    cv::LUT(input_image, lut_matrix, gamma_corrected_image);
    return gamma_corrected_image;
}

// Function to blend two images with specified weights
cv::Mat blend_images(const cv::Mat& image1, const cv::Mat& image2, double alpha) {
    cv::Mat blended_image;
    cv::addWeighted(image1, alpha, image2, 1.0 - alpha, 0.0, blended_image);
    return blended_image;
}

// Function to calculate the Structural Similarity Index (SSIM) between two images
double calculate_ssim(const cv::Mat& image1, const cv::Mat& image2) {
    cv::Mat gray1, gray2;
    cv::cvtColor(image1, gray1, cv::COLOR_BGR2GRAY);
    cv::cvtColor(image2, gray2, cv::COLOR_BGR2GRAY);
    
    cv::Mat ssim_map;
    double C1 = 6.5025, C2 = 58.5225;
    cv::Mat I1_2 = gray1.mul(gray1);
    cv::Mat I2_2 = gray2.mul(gray2);
    cv::Mat I1_I2 = gray1.mul(gray2);

    cv::Mat mu1, mu2;
    cv::GaussianBlur(gray1, mu1, cv::Size(11, 11), 1.5);
    cv::GaussianBlur(gray2, mu2, cv::Size(11, 11), 1.5);

    cv::Mat mu1_2 = mu1.mul(mu1);
    cv::Mat mu2_2 = mu2.mul(mu2);
    cv::Mat mu1_mu2 = mu1.mul(mu2);

    cv::Mat sigma1_2, sigma2_2, sigma12;
    cv::GaussianBlur(I1_2, sigma1_2, cv::Size(11, 11), 1.5);
    sigma1_2 -= mu1_2;
    cv::GaussianBlur(I2_2, sigma2_2, cv::Size(11, 11), 1.5);
    sigma2_2 -= mu2_2;
    cv::GaussianBlur(I1_I2, sigma12, cv::Size(11, 11), 1.5);
    sigma12 -= mu1_mu2;

    cv::Mat t1, t2, t3;
    t1 = 2 * mu1_mu2 + C1;
    t2 = 2 * sigma12 + C2;
    t3 = t1.mul(t2);

    t1 = mu1_2 + mu2_2 + C1;
    t2 = sigma1_2 + sigma2_2 + C2;
    t1 = t1.mul(t2);

    cv::Mat ssim_map = t3 / t1;
    return cv::mean(ssim_map)[0];
}

// Function to draw bounding boxes on an image
cv::Mat draw_bounding_boxes(const cv::Mat& input_image, const std::vector<cv::Rect>& bounding_boxes) {
    cv::Mat image_with_boxes = input_image.clone();
    for (const auto& box : bounding_boxes) {
        cv::rectangle(image_with_boxes, box, cv::Scalar(0, 255, 0), 2);
    }
    return image_with_boxes;
}

// Function to find and draw circles in an image using the Hough Transform
cv::Mat detect_and_draw_circles(const cv::Mat& input_image, double dp, double min_dist, double param1, double param2, int min_radius, int max_radius) {
    cv::Mat gray_image = convert_to_grayscale(input_image);
    std::vector<cv::Vec3f> circles;
    cv::HoughCircles(gray_image, circles, cv::HOUGH_GRADIENT, dp, min_dist, param1, param2, min_radius, max_radius);
    cv::Mat output_image = input_image.clone();
    for (size_t i = 0; i < circles.size(); i++) {
        cv::Point center(cvRound(circles[i][0]), cvRound(circles[i][1]));
        int radius = cvRound(circles[i][2]);
        cv::circle(output_image, center, radius, cv::Scalar(0, 255, 0), 2);
    }
    return output_image;
}

// Function to draw text on an image
cv::Mat draw_text(const cv::Mat& input_image, const std::string& text, const cv::Point& position, int font, double scale, const cv::Scalar& color, int thickness) {
    cv::Mat image_with_text = input_image.clone();
    cv::putText(image_with_text, text, position, font, scale, color, thickness);
    return image_with_text;
}

// Function to apply histogram equalization for color images (LAB color space)
cv::Mat equalize_color_histogram(const cv::Mat& input_image) {
    cv::Mat lab_image;
    cv::cvtColor(input_image, lab_image, cv::COLOR_BGR2Lab);

    std::vector<cv::Mat> lab_planes(3);
    cv::split(lab_image, lab_planes);
    cv::equalizeHist(lab_planes[0], lab_planes[0]);

    cv::Mat equalized_image;
    cv::merge(lab_planes, equalized_image);
    cv::cvtColor(equalized_image, equalized_image, cv::COLOR_Lab2BGR);

    return equalized_image;
}

// Function to detect lines in an image using Hough Line Transform
cv::Mat detect_and_draw_lines(const cv::Mat& input_image, double rho, double theta, int threshold) {
    cv::Mat gray_image = convert_to_grayscale(input_image);
    std::vector<cv::Vec2f> lines;
    cv::HoughLines(gray_image, lines, rho, theta, threshold);
    cv::Mat output_image = input_image.clone();
    for (size_t i = 0; i < lines.size(); i++) {
        float rho = lines[i][0], theta = lines[i][1];
        double a = cos(theta), b = sin(theta);
        double x0 = a * rho, y0 = b * rho;
        cv::Point pt1(cvRound(x0 + 1000 * (-b)), cvRound(y0 + 1000 * (a)));
        cv::Point pt2(cvRound(x0 - 1000 * (-b)), cvRound(y0 - 1000 * (a)));
        cv::line(output_image, pt1, pt2, cv::Scalar(0, 255, 0), 2);
    }
    return output_image;
}

// Function to detect and draw polygons in an image
cv::Mat detect_and_draw_polygons(const cv::Mat& input_image, double epsilon, bool closed) {
    cv::Mat gray_image = convert_to_grayscale(input_image);
    cv::Mat edges = detect_edges(gray_image, 50, 150);
    std::vector<std::vector<cv::Point>> contours;
    cv::findContours(edges, contours, cv::RETR_TREE, cv::CHAIN_APPROX_SIMPLE);

    cv::Mat output_image = input_image.clone();
    for (size_t i = 0; i < contours.size(); i++) {
        std::vector<cv::Point> approx;
        cv::approxPolyDP(contours[i], approx, epsilon, closed);
        cv::polylines(output_image, approx, true, cv::Scalar(0, 255, 0), 2);
    }
    return output_image;
}

// Function to invert the colors of an image
cv::Mat invert_colors(const cv::Mat& input_image) {
    cv::Mat inverted_image;
    cv::bitwise_not(input_image, inverted_image);
    return inverted_image;
}

// Function to overlay a semi-transparent mask on an image
cv::Mat overlay_mask(const cv::Mat& input_image, const cv::Mat& mask, const cv::Scalar& color, double alpha) {
    cv::Mat colored_mask(input_image.size(), input_image.type(), color);
    cv::Mat result;
    cv::addWeighted(input_image, 1.0, colored_mask, alpha, 0.0, result);
    return result;
}

// Function to detect corners in an image using Harris corner detection
cv::Mat detect_corners(const cv::Mat& input_image, double block_size, double aperture_size, double k) {
    cv::Mat grayscale_image = convert_to_grayscale(input_image);
    cv::Mat corners, corners_normalized;
    cv::cornerHarris(grayscale_image, corners, block_size, aperture_size, k);
    cv::normalize(corners, corners_normalized, 0, 255, cv::NORM_MINMAX, CV_32FC1, cv::Mat());

    cv::Mat output_image = input_image.clone();
    for (int i = 0; i < corners_normalized.rows; i++) {
        for (int j = 0; j < corners_normalized.cols; j++) {
            if ((int)corners_normalized.at<float>(i, j) > 200) {
                cv::circle(output_image, cv::Point(j, i), 5, cv::Scalar(0, 255, 0), 2);
            }
        }
    }
    return output_image;
}

// Function to detect keypoints using ORB (Oriented FAST and Rotated BRIEF)
cv::Mat detect_keypoints_orb(const cv::Mat& input_image) {
    cv::Ptr<cv::ORB> orb = cv::ORB::create();
    std::vector<cv::KeyPoint> keypoints;
    cv::Mat descriptors;

    orb->detectAndCompute(input_image, cv::noArray(), keypoints, descriptors);
    cv::Mat output_image = input_image.clone();
    cv::drawKeypoints(input_image, keypoints, output_image, cv::Scalar(0, 255, 0), cv::DrawMatchesFlags::DRAW_RICH_KEYPOINTS);

    return output_image;
}

// Function to detect and match keypoints between two images using ORB
cv::Mat match_keypoints_orb(const cv::Mat& image1, const cv::Mat& image2) {
    cv::Ptr<cv::ORB> orb = cv::ORB::create();
    std::vector<cv::KeyPoint> keypoints1, keypoints2;
    cv::Mat descriptors1, descriptors2;

    orb->detectAndCompute(image1, cv::noArray(), keypoints1, descriptors1);
    orb->detectAndCompute(image2, cv::noArray(), keypoints2, descriptors2);

    cv::BFMatcher matcher(cv::NORM_HAMMING, true);
    std::vector<cv::DMatch> matches;
    matcher.match(descriptors1, descriptors2, matches);

    cv::Mat matched_image;
    cv::drawMatches(image1, keypoints1, image2, keypoints2, matches, matched_image);

    return matched_image;
}

// Function to detect faces in an image using a pre-trained Haar Cascade classifier
cv::Mat detect_faces(const cv::Mat& input_image, const std::string& cascade_path) {
    cv::CascadeClassifier face_cascade;
    if (!face_cascade.load(cascade_path)) {
        std::cerr << "Error: Could not load face cascade from " << cascade_path << std::endl;
        return input_image;
    }

    std::vector<cv::Rect> faces;
    cv::Mat grayscale_image = convert_to_grayscale(input_image);
    face_cascade.detectMultiScale(grayscale_image, faces, 1.1, 3, 0, cv::Size(30, 30));

    cv::Mat output_image = input_image.clone();
    for (size_t i = 0; i < faces.size(); i++) {
        cv::rectangle(output_image, faces[i], cv::Scalar(255, 0, 0), 2);
    }

    return output_image;
}

// Function to detect eyes in an image using a pre-trained Haar Cascade classifier
cv::Mat detect_eyes(const cv::Mat& input_image, const std::string& cascade_path) {
    cv::CascadeClassifier eye_cascade;
    if (!eye_cascade.load(cascade_path)) {
        std::cerr << "Error: Could not load eye cascade from " << cascade_path << std::endl;
        return input_image;
    }

    std::vector<cv::Rect> eyes;
    cv::Mat grayscale_image = convert_to_grayscale(input_image);
    eye_cascade.detectMultiScale(grayscale_image, eyes, 1.1, 3, 0, cv::Size(30, 30));

    cv::Mat output_image = input_image.clone();
    for (size_t i = 0; i < eyes.size(); i++) {
        cv::rectangle(output_image, eyes[i], cv::Scalar(0, 255, 0), 2);
    }

    return output_image;
}

// Function to apply a perspective transformation to an image
cv::Mat apply_perspective_transform(const cv::Mat& input_image, const std::vector<cv::Point2f>& source_points, const std::vector<cv::Point2f>& dest_points) {
    cv::Mat perspective_matrix = cv::getPerspectiveTransform(source_points, dest_points);
    cv::Mat transformed_image;
    cv::warpPerspective(input_image, transformed_image, perspective_matrix, input_image.size());
    return transformed_image;
}

// Function to resize the image using nearest neighbor interpolation
cv::Mat resize_nearest_neighbor(const cv::Mat& input_image, int target_width, int target_height) {
    cv::Mat resized_image;
    cv::resize(input_image, resized_image, cv::Size(target_width, target_height), 0, 0, cv::INTER_NEAREST);
    return resized_image;
}

// Function to resize the image using cubic interpolation
cv::Mat resize_cubic(const cv::Mat& input_image, int target_width, int target_height) {
    cv::Mat resized_image;
    cv::resize(input_image, resized_image, cv::Size(target_width, target_height), 0, 0, cv::INTER_CUBIC);
    return resized_image;
}

// Function to detect blobs in an image
cv::Mat detect_blobs(const cv::Mat& input_image) {
    cv::Ptr<cv::SimpleBlobDetector> detector = cv::SimpleBlobDetector::create();
    std::vector<cv::KeyPoint> keypoints;
    detector->detect(input_image, keypoints);

    cv::Mat blob_image;
    cv::drawKeypoints(input_image, keypoints, blob_image, cv::Scalar(0, 0, 255), cv::DrawMatchesFlags::DRAW_RICH_KEYPOINTS);
    return blob_image;
}

// Function to apply dilation to an image
cv::Mat apply_dilation(const cv::Mat& input_image, int dilation_size) {
    cv::Mat dilated_image;
    cv::Mat element = cv::getStructuringElement(cv::MORPH_RECT, cv::Size(2 * dilation_size + 1, 2 * dilation_size + 1));
    cv::dilate(input_image, dilated_image, element);
    return dilated_image;
}

// Function to apply erosion to an image
cv::Mat apply_erosion(const cv::Mat& input_image, int erosion_size) {
    cv::Mat eroded_image;
    cv::Mat element = cv::getStructuringElement(cv::MORPH_RECT, cv::Size(2 * erosion_size + 1, 2 * erosion_size + 1));
    cv::erode(input_image, eroded_image, element);
    return eroded_image;
}

// Function to apply opening (erosion followed by dilation)
cv::Mat apply_opening(const cv::Mat& input_image, int kernel_size) {
    cv::Mat opened_image;
    cv::Mat element = cv::getStructuringElement(cv::MORPH_RECT, cv::Size(kernel_size, kernel_size));
    cv::morphologyEx(input_image, opened_image, cv::MORPH_OPEN, element);
    return opened_image;
}

// Function to apply closing (dilation followed by erosion)
cv::Mat apply_closing(const cv::Mat& input_image, int kernel_size) {
    cv::Mat closed_image;
    cv::Mat element = cv::getStructuringElement(cv::MORPH_RECT, cv::Size(kernel_size, kernel_size));
    cv::morphologyEx(input_image, closed_image, cv::MORPH_CLOSE, element);
    return closed_image;
}

// Function to apply morphological gradient (difference between dilation and erosion)
cv::Mat apply_morph_gradient(const cv::Mat& input_image, int kernel_size) {
    cv::Mat gradient_image;
    cv::Mat element = cv::getStructuringElement(cv::MORPH_RECT, cv::Size(kernel_size, kernel_size));
    cv::morphologyEx(input_image, gradient_image, cv::MORPH_GRADIENT, element);
    return gradient_image;
}

// Function to apply top-hat transformation (difference between input image and opening)
cv::Mat apply_top_hat(const cv::Mat& input_image, int kernel_size) {
    cv::Mat top_hat_image;
    cv::Mat element = cv::getStructuringElement(cv::MORPH_RECT, cv::Size(kernel_size, kernel_size));
    cv::morphologyEx(input_image, top_hat_image, cv::MORPH_TOPHAT, element);
    return top_hat_image;
}

// Function to apply black-hat transformation (difference between closing and input image)
cv::Mat apply_black_hat(const cv::Mat& input_image, int kernel_size) {
    cv::Mat black_hat_image;
    cv::Mat element = cv::getStructuringElement(cv::MORPH_RECT, cv::Size(kernel_size, kernel_size));
    cv::morphologyEx(input_image, black_hat_image, cv::MORPH_BLACKHAT, element);
    return black_hat_image;
}

// Function to calculate the Fast Fourier Transform (FFT) of an image
cv::Mat calculate_fft(const cv::Mat& input_image) {
    cv::Mat gray_image = convert_to_grayscale(input_image);
    cv::Mat padded;
    int m = cv::getOptimalDFTSize(gray_image.rows);
    int n = cv::getOptimalDFTSize(gray_image.cols);
    cv::copyMakeBorder(gray_image, padded, 0, m - gray_image.rows, 0, n - gray_image.cols, cv::BORDER_CONSTANT, cv::Scalar::all(0));

    cv::Mat planes[] = {cv::Mat_<float>(padded), cv::Mat::zeros(padded.size(), CV_32F)};
    cv::Mat complex_image;
    cv::merge(planes, 2, complex_image);

    cv::dft(complex_image, complex_image);

    cv::split(complex_image, planes);
    cv::magnitude(planes[0], planes[1], planes[0]);
    cv::Mat magnitude_image = planes[0];

    magnitude_image += cv::Scalar::all(1);
    cv::log(magnitude_image, magnitude_image);

    magnitude_image = magnitude_image(cv::Rect(0, 0, magnitude_image.cols & -2, magnitude_image.rows & -2));
    int cx = magnitude_image.cols / 2;
    int cy = magnitude_image.rows / 2;

    cv::Mat q0(magnitude_image, cv::Rect(0, 0, cx, cy));
    cv::Mat q1(magnitude_image, cv::Rect(cx, 0, cx, cy));
    cv::Mat q2(magnitude_image, cv::Rect(0, cy, cx, cy));
    cv::Mat q3(magnitude_image, cv::Rect(cx, cy, cx, cy));

    cv::Mat tmp;
    q0.copyTo(tmp);
    q3.copyTo(q0);
    tmp.copyTo(q3);
    q1.copyTo(tmp);
    q2.copyTo(q1);
    tmp.copyTo(q2);

    cv::normalize(magnitude_image, magnitude_image, 0, 1, cv::NORM_MINMAX);
    return magnitude_image;
}