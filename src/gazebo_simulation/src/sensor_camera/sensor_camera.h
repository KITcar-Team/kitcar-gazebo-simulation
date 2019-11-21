#ifndef SENSOR_CAMERA_H
#define SENSOR_CAMERA_H
#include <common/macros.h>

THIRD_PARTY_HEADERS_BEGIN

#include <message_filters/subscriber.h>
#include <message_filters/time_synchronizer.h>

#include <opencv2/core/core.hpp>
#include <opencv2/imgproc/imgproc.hpp>
THIRD_PARTY_HEADERS_END

#include "common/parameter_interface.h"

/*!
 * \brief Precrops camera image outputted by Gazebo
 */
class SensorCamera {
 public:
  /*!
   * \brief SensorCamera is the consstructor. A ros indipendent functionality
   * containing
   * class needs a pointer to a ParameterInterface (in fact a ParameterHandler)
   * to get access to parameters.
   * \param parameters the ParameterInterface
   */
  SensorCamera(ParameterInterface* parameters);

  /*!
   * \brief precropImage
   *
   * performs image cropping
   *
   * @param image_gray the source image
   * @param preprocessed_image the output image
   */
  void precropImage(const cv::Mat& image_uncropped, cv::Mat& image_cropped);

  cv::Rect image_limits;

 private:
  static const ParameterString<int> OUTPUT_START_X;
  static const ParameterString<int> OUTPUT_END_X;
  static const ParameterString<int> OUTPUT_START_Y;
  static const ParameterString<int> OUTPUT_END_Y;

  static const ParameterString<int> NOISE_TYPE;
  static const ParameterString<int> MEAN_VALUE;
  static const ParameterString<int> STANDARD_DEVIATION;

  /*!
   * \brief parameters_ptr_ is needed for parameter access
   */
  ParameterInterface* parameters_ptr_;

  void gaussianNoise(const cv::Mat& image);

  void saltPepperNoise(const cv::Mat& image);
};

#endif  // SENSOR_CAMERA_H
