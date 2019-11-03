#ifndef SENSOR_TOF_H
#define SENSOR_TOF_H

#include <common/macros.h>

THIRD_PARTY_HEADERS_BEGIN

#include <sensor_msgs/PointCloud2.h>

THIRD_PARTY_HEADERS_END

#include "common/parameter_interface.h"

/*!
 * \brief Used to publish time of flight sensors data in simulation
 */
class SensorTof {
 public:
  /*!
  * \brief SensorTof is the consstructor. A ros indipendent
  * functionality containing
  * class needs a pointer to a ParameterInterface (in fact a ParameterHandler)
  * to get access to parameters.
  * \param parameters the ParameterInterface
  */
  SensorTof(ParameterInterface *parameters);

  /*!
   * \brief preprocessImage
   *
   * performs image cropping
   *
   * @param image_gray the source image
   * @param preprocessed_image the output image
   */
  float processPointCloud(const sensor_msgs::PointCloud2 cloud);

 private:
  /*!
  * \brief parameters_ptr_ is needed for parameter access
  */
  ParameterInterface *parameters_ptr_;
};

#endif  // SENSOR_TOF_H
