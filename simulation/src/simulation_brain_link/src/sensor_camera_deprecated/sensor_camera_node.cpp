#include "sensor_camera_node.h"

#include <cv_bridge/cv_bridge.h>
#include <sensor_msgs/image_encodings.h>
#include <opencv2/core.hpp>
#include <thread>

SensorCameraNode::SensorCameraNode(ros::NodeHandle &node_handle)
    : node_handle_(node_handle), sensor_camera_(node_handle_) {}

void SensorCameraNode::startModule() {

  ROS_INFO("Starting module SensorCameraNode");

  // Read sub and pub topics.
  std::string sub_topic, pub_topic;

  node_handle_.param<std::string>(
      "pub_topic", pub_topic, "/simulation/sensors/prepared/camera");
  node_handle_.param<std::string>(
      "sub_topic", sub_topic, "/simulation/sensors/raw/camera");

  rospub_image = node_handle_.advertise<sensor_msgs::Image>(pub_topic, 1);

  image_transport::ImageTransport img_trans(node_handle_);
  rossub_uncropped_image =
      img_trans.subscribe(sub_topic, 1, &SensorCameraNode::handleImage, this);
}

void SensorCameraNode::stopModule() {
  rossub_uncropped_image.shutdown();
  rospub_image.shutdown();
}

void SensorCameraNode::handleImage(const sensor_msgs::ImageConstPtr &msg) {

  cv_bridge::CvImageConstPtr cv_ptr;
  try {
    cv_ptr = cv_bridge::toCvShare(msg, sensor_msgs::image_encodings::MONO8);
  } catch (const cv_bridge::Exception &e) {
    ROS_ERROR("cv_bridge exception: %s", e.what());
    return;
  }
  cv_bridge::CvImage out_msg;
  out_msg.header = msg->header;
  out_msg.encoding = sensor_msgs::image_encodings::MONO8;

  sensor_camera_.precropImage(cv_ptr->image, out_msg.image);

  sensor_msgs::ImagePtr out_ptr(out_msg.toImageMsg());
  rospub_image.publish(out_ptr);
}


std::string SensorCameraNode::getName() { return std::string("sensor_camera"); }

int main(int argc, char *argv[]) {
  // Start, loop and stop node!
  ros::init(argc, argv, SensorCameraNode::getName());
  ros::NodeHandle nh("~");
  SensorCameraNode node(nh);
  node.startModule();
  while (ros::ok()) {
    ros::spinOnce();
  }
  node.stopModule();
  return 0;
}
