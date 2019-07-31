#include "segmentation.h"

Segmentation::Segmentation(ParameterInterface* parameters)
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

void Segmentation::modifyImage(const cv::Mat& image_uncropped, cv::Mat& image_cropped) {
  // cutting out region of interest
  image_cropped = image_uncropped(image_limits);
}

void Segmentation::storeImages(cv::Mat default_image, cv::Mat segmentation_image){

  std::string path = "/home/ditschuk/kitcar/kitcar-gazebo-simulation/datasets/segmentation/";

  cv::imwrite(path + "test.png",default_image);

  cv::imwrite(path + "test_segmented.png",segmentation_image);

  cv::namedWindow("image", cv::WINDOW_AUTOSIZE);
  cv::imshow("image", default_image);
  cv::waitKey(30);
}
