#ifndef SEGMENTATION_H
#define SEGMENTATION_H
#include <common/macros.h>

THIRD_PARTY_HEADERS_BEGIN

#include <message_filters/subscriber.h>
#include <message_filters/time_synchronizer.h>

#include <opencv2/core/core.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/opencv.hpp>

THIRD_PARTY_HEADERS_END

#include "common/parameter_interface.h"

/*!
 * \brief Precrops camera image outputted by Gazebo
 */
class Segmentation {
 public:
  /*!
   * \brief PreCropping is the consstructor. A ros indipendent functionality
   * containing
   * class needs a pointer to a ParameterInterface (in fact a ParameterHandler)
   * to get access to parameters.
   * \param parameters the ParameterInterface
   */
  Segmentation(ParameterInterface* parameters);

  /*!
   * \brief precropImage
   *
   * performs image cropping
   *
   * @param image_gray the source image
   * @param preprocessed_image the output image
   */
  void modifyImage(const cv::Mat& image_uncropped, cv::Mat& image_cropped);

  void storeImages(cv::Mat default_image, cv::Mat segmentation_image);


 private:
  /*!
   * \brief ROI_START_LOC_X
   *
   * Start location (x-axis) of the ROI area relative to recorded image size.
   */
  static const ParameterString<int> ROI_START_LOC_X;
  /*!
   * \brief ROI_START_LOC_Y
   *
   * Start location (y-axis) of the ROI area relative to recorded image size.
   */
  static const ParameterString<int> ROI_START_LOC_Y;
  /*!
   * \brief ROI_END_LOC_X
   *
   * End location (x-axis) of the ROI area relative to recorded image size.
   */
  static const ParameterString<int> ROI_END_LOC_X;
  /*!
   * \brief ROI_END_LOC_Y
   *
   * End location (y-axis) of the ROI area relative to recorded image size.
   */
  static const ParameterString<int> ROI_END_LOC_Y;


  /*!
   * \brief parameters_ptr_ is needed for parameter access
   */
  ParameterInterface* parameters_ptr_;

  cv::Rect image_limits;
};

#endif  // SEGMENTATION_H
