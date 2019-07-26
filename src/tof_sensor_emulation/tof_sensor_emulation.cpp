#include "tof_sensor_emulation.h"

THIRD_PARTY_HEADERS_BEGIN
#include <sensor_msgs/PointCloud2.h>

#include <pcl/point_types.h>

#include <pcl_ros/point_cloud.h>

THIRD_PARTY_HEADERS_END

TofSensorEmulation::TofSensorEmulation(ParameterInterface* parameters)
    : parameters_ptr_(parameters) {}

float TofSensorEmulation::processPointCloud(const sensor_msgs::PointCloud2 cloud) {

  pcl::PointCloud<pcl::PointXYZ> c;
  pcl::fromROSMsg(cloud, c);

  float minDis = 2;

  uint32_t rows = c.height;
  for (uint32_t i = 0; i < c.width; i++) {
    float r = c.at(int(i), rows / 2).getVector3fMap().squaredNorm();
    if (r < minDis) {
      minDis = r;
    }
  }
  return minDis;
}
