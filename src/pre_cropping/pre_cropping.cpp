#include "pre_cropping.h"

PreCropping::PreCropping(ParameterInterface* parameters)
    : parameters_ptr_(parameters) {

  /*image_limits.x = parameters_ptr_->getParam(ROI_START_LOC_X);
  image_limits.y = parameters_ptr_->getParam(ROI_START_LOC_Y);
  image_limits.width = parameters_ptr_->getParam(ROI_END_LOC_X) -
  image_limits.x;
  image_limits.height = parameters_ptr_->getParam(ROI_END_LOC_Y) -
  image_limits.y;*/
  image_limits.x = 0;
  image_limits.y = 1024 - 650;
  image_limits.width = 1280 - image_limits.x;
  image_limits.height = 1024 - image_limits.y;
}

void PreCropping::precropImage(const cv::Mat& image_uncropped, cv::Mat& image_cropped) {
  // cutting out region of interest
  image_cropped = image_uncropped(image_limits);
}
