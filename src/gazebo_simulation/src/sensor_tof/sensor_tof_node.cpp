#include "sensor_tof_node.h"

THIRD_PARTY_HEADERS_BEGIN
#include <cv_bridge/cv_bridge.h>
#include <sensor_msgs/PointCloud2.h>
#include <sensor_msgs/Range.h>

#include <std_msgs/Empty.h>

#include <thread>


THIRD_PARTY_HEADERS_END

#include "common/node_creation_makros.h"

SensorTofNode::SensorTofNode(ros::NodeHandle &node_handle)
    : NodeBase(node_handle), sensor_tof_(&parameter_handler_) {

  node_handle.param<std::string>(
      "pub_topic",
      pub_topic, 
			"/simulation/sensors/prepared/tof");
  node_handle.param<std::string>(
      "sub_topic", sub_topic,
			"/simulation/sensors/raw/tof");
  node_handle.param<std::string>(
      "frame_id", frame_id);

  ROS_INFO("Starting SensorTofNode sub %s and pub %s ",
           sub_topic.c_str(),
           pub_topic.c_str());
}


void SensorTofNode::startModule() {

  ROS_INFO("Starting module SensorTofNode and subscribing to %s",
           sub_topic.c_str());


  // sets your node in running mode. Activate publishers, subscribers, service
  // servers, etc here.
  rospub_tof = node_handle_.advertise<sensor_msgs::Range>(pub_topic, 1);

  rossub_raw_depth = node_handle_.subscribe<sensor_msgs::PointCloud2>(
      sub_topic, 1, &SensorTofNode::handleRawDepthImage, this);
}

void SensorTofNode::stopModule() {
  rospub_tof.shutdown();
  rossub_raw_depth.shutdown();
}

void SensorTofNode::handleRawDepthImage(const sensor_msgs::PointCloud2ConstPtr &msg) {

  sensor_msgs::Range out_msg;

  //! @todo add this sensor parameters (enables correct visualization)
  out_msg.field_of_view = static_cast<float>(M_PI_4);
  out_msg.min_range = 0.01f;
  out_msg.max_range = 2.0f;

  out_msg.header.frame_id = frame_id;

  out_msg.range = sensor_tof_.processPointCloud(*msg);

  rospub_tof.publish(out_msg);
}



std::string SensorTofNode::getName() {
  return std::string("sensor_tof");
}

CREATE_NODE(SensorTofNode)
