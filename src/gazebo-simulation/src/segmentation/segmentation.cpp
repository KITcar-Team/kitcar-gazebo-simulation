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

  int time = int(ros::Time::now().toSec() * 10);

  std::string file = std::to_string(time);

  cv::imwrite(path + "data/" + file + ".png",default_image);

  cv::imwrite(path + "labels/" +  file + ".png",segmentation_image);
}


std::string  Segmentation::genRandom(uint max_length, std::string char_index)
{ // maxLength and charIndex can be customized, but I've also initialized them.
    uint length = rand() % max_length + 1; // length of the string is a random value that can be up to 'l' characters.

    uint indexesOfRandomChars[15]; // array of random values that will be used to iterate through random indexes of 'charIndex'
    for (uint i = 0; i < length; ++i) // assigns a random number to each index of "indexesOfRandomChars"
        indexesOfRandomChars[i] = rand() % char_index.length();

    std::string randomString = ""; // random string that will be returned by this function
    for (uint i = 0; i < length; ++i)// appends a random amount of random characters to "randomString"
    {
        randomString += char_index[indexesOfRandomChars[i]];
    }
    return randomString;
}
