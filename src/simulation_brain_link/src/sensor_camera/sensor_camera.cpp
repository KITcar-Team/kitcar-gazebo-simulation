#include "sensor_camera.h"

// Define parameters for image dimensions
const ParameterString<int> SensorCamera::OUTPUT_START_X("output_start_x");
const ParameterString<int> SensorCamera::OUTPUT_END_X("output_end_x");
const ParameterString<int> SensorCamera::OUTPUT_START_Y("output_start_y");
const ParameterString<int> SensorCamera::OUTPUT_END_Y("output_end_y");

// Define parameters for noise
const ParameterString<int> SensorCamera::NOISE_TYPE("noise_type");
const ParameterString<int> SensorCamera::MEAN_VALUE("mean_value");
const ParameterString<int> SensorCamera::STANDARD_DEVIATION(
    "standard_deviation");
const ParameterString<int> SensorCamera::STEP("step");


SensorCamera::SensorCamera(ParameterInterface* parameters)
    : parameters_ptr_(parameters) {

  // Register parameters for image dimensions
  parameters_ptr_->registerParam(OUTPUT_START_X);
  parameters_ptr_->registerParam(OUTPUT_END_X);
  parameters_ptr_->registerParam(OUTPUT_START_Y);
  parameters_ptr_->registerParam(OUTPUT_END_Y);

  // Register parameters for noise
  parameters_ptr_->registerParam(NOISE_TYPE);
  parameters_ptr_->registerParam(MEAN_VALUE);
  parameters_ptr_->registerParam(STANDARD_DEVIATION);
  parameters_ptr_->registerParam(STEP);


  // Get parameters for image dimensions
  image_limits.x = parameters_ptr_->getParam(OUTPUT_START_X);
  image_limits.y = parameters_ptr_->getParam(OUTPUT_START_Y);
  image_limits.width = parameters_ptr_->getParam(OUTPUT_END_X) - image_limits.x;
  image_limits.height = parameters_ptr_->getParam(OUTPUT_END_Y) - image_limits.y;
}

void SensorCamera::precropImage(const cv::Mat& image_uncropped, cv::Mat& image_cropped) {
  // cutting out region of interest
  image_cropped = image_uncropped(image_limits);

  int noise_type = parameters_ptr_->getParam(NOISE_TYPE);

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
  int mean_value = parameters_ptr_->getParam(MEAN_VALUE);
  int standard_deviation = parameters_ptr_->getParam(STANDARD_DEVIATION);

  // Create noise matrix (height x width from image)
  cv::Mat mat_noise = cv::Mat(image_limits.size(), CV_16S);
  cv::randn(mat_noise, mean_value, standard_deviation);

  cv::addWeighted(image, 1.0, mat_noise, 1.0, 0.0, image);  // Add noise to image
}

void SensorCamera::saltPepperNoise(const cv::Mat& image) {
  int step = parameters_ptr_->getParam(STEP);

  // Loop over image
  cv::Mat img = image;
  int rand_idx = rand() % step;
  while (rand_idx < image.rows * image.cols) {
    img.at<uchar>(rand_idx / image.cols, rand_idx % image.cols) = (rand() % 2) * 255;
    rand_idx += rand() % step;
  }
}
