#include "sensor_camera.h"


SensorCamera::SensorCamera(ros::NodeHandle nh) : node_handle_(nh) {

  int start_x, start_y, end_x, end_y;
  // Get parameters for image dimensions
  nh.getParam("output_start_x", start_x);
  nh.getParam("output_start_y", start_y);
  nh.getParam("output_end_x", end_x);
  nh.getParam("output_end_y", end_y);

  image_limits.x = start_x;
  image_limits.y = start_y;
  image_limits.width = end_x - image_limits.x;
  image_limits.height = end_y - image_limits.y;
}

void SensorCamera::precropImage(const cv::Mat& image_uncropped, cv::Mat& image_cropped) {
  // cutting out region of interest
  image_cropped = image_uncropped(image_limits);

  int noise_type;
  node_handle_.getParam("noise_type", noise_type);

  switch (noise_type) {
    case 1:
      gaussianNoise(image_cropped);
      break;
    case 2:
      saltPepperNoise(image_cropped);
      break;

    default:
      break;
  }
}

void SensorCamera::gaussianNoise(const cv::Mat& image) {
  // Get parameters for noise
  int mean_value, standard_deviation;
  node_handle_.getParam("mean_value", mean_value);
  node_handle_.getParam("standard_deviation", standard_deviation);

  // Create noise matrix (height x width from image)
  cv::Mat mat_noise = cv::Mat(image_limits.size(), CV_16S);
  cv::randn(mat_noise, mean_value, standard_deviation);

  cv::addWeighted(image, 1.0, mat_noise, 1.0, 0.0, image);  // Add noise to image
}

void SensorCamera::saltPepperNoise(const cv::Mat& image) {
  int step;
  node_handle_.getParam("step", step);

  // Loop over image
  cv::Mat img = image;
  int rand_idx = rand() % step;
  while (rand_idx < image.rows * image.cols) {
    img.at<uchar>(rand_idx / image.cols, rand_idx % image.cols) = (rand() % 2) * 255;
    rand_idx += rand() % step;
  }
}
