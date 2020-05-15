#ifndef SENSOR_CAMERA_H
#define SENSOR_CAMERA_H

#include <message_filters/subscriber.h>
#include <message_filters/time_synchronizer.h>

#include <opencv2/core/core.hpp>
#include <opencv2/imgproc/imgproc.hpp>

#include "ros/ros.h"

/*!
 * \brief Precrops camera image outputted by Gazebo
 */
class SensorCamera {
 public:
  /*!
   * \brief SensorCamera is the constructor. A ros independent functionality
   * containing
   * \param nh node handle to access parameters
   */
  SensorCamera(ros::NodeHandle nh);

  /*!
   * \brief precropImage
   *
   * Crop the image to the region of interest.
   * Optionally (depending on ROS parameter "noise_type" noise can be applied.
   *
   * \param image_uncropped the source image
   * \param image_cropped the output image
   */
  void precropImage(const cv::Mat& image_uncropped, cv::Mat& image_cropped);

  /*!
   * \brief image_limits region of interest that the image is cropped to.
   */
  cv::Rect image_limits;

 private:
  /*!
   * \brief node_handle_ is needed for parameter access
   */
  ros::NodeHandle node_handle_;


  /*!
   * \brief gaussianNoise
   *
   * Apply gaussian noise to the image
   *
   * \param image
   */
  void gaussianNoise(const cv::Mat& image);

  /*!
   * \brief saltPepperNoise
   *
   * Apply salt and pepper noise to the image
   * Slow due to poor implementation!
   *
   * \param image
   */
  void saltPepperNoise(const cv::Mat& image);
};

#endif  // SENSOR_CAMERA_H
