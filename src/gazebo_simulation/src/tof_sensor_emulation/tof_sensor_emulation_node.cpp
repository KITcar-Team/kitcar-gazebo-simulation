#include "tof_sensor_emulation_node.h"

THIRD_PARTY_HEADERS_BEGIN
#include <cv_bridge/cv_bridge.h>
#include <sensor_msgs/PointCloud2.h>
#include <sensor_msgs/Range.h>

#include <std_msgs/Empty.h>

#include <thread>


THIRD_PARTY_HEADERS_END

#include "common/node_creation_makros.h"

TofSensorEmulationNode::TofSensorEmulationNode(ros::NodeHandle &node_handle)
    : NodeBase(node_handle), tof_sensor_emulation_(&parameter_handler_) {

  node_handle.param<std::string>(
      "pub_topic",
      pub_topic,
      "/controller_interface/controller_interface/infrared_sensor_back");
  node_handle.param<std::string>(
      "sub_topic", sub_topic, "/camera/depth_raw/back");
  node_handle.param<std::string>(
      "frame_id", frame_id, "ir_back");

  ROS_INFO("Starting TofSensorEmulationNode sub %s and pub %s ",
           sub_topic.c_str(),
           pub_topic.c_str());
}


void TofSensorEmulationNode::startModule() {

  ROS_INFO("Starting module TofSensorEmulationNode and subscribing to %s",
           sub_topic.c_str());


  // sets your node in running mode. Activate publishers, subscribers, service
  // servers, etc here.
  rospub_tof = node_handle_.advertise<sensor_msgs::Range>(pub_topic, 1);

  rossub_raw_depth = node_handle_.subscribe<sensor_msgs::PointCloud2>(
      sub_topic, 1, &TofSensorEmulationNode::handleRawDepthImage, this);
}

void TofSensorEmulationNode::stopModule() {
  rospub_tof.shutdown();
  rossub_raw_depth.shutdown();
}

void TofSensorEmulationNode::handleRawDepthImage(const sensor_msgs::PointCloud2ConstPtr &msg) {

  sensor_msgs::Range out_msg;

  //! @todo add this sensor parameters (enables correct visualization)
  out_msg.field_of_view = static_cast<float>(M_PI_4);
  out_msg.min_range = 0.01f;
  out_msg.max_range = 2.0f;

  out_msg.header.frame_id = frame_id;

  out_msg.range = tof_sensor_emulation_.processPointCloud(*msg);

  rospub_tof.publish(out_msg);
}



std::string TofSensorEmulationNode::getName() {
  return std::string("tof_sensor_emulation");
}

CREATE_NODE(TofSensorEmulationNode)
