#include "sensor_camera_node.h"

THIRD_PARTY_HEADERS_BEGIN
#include <cv_bridge/cv_bridge.h>
#include <sensor_msgs/image_encodings.h>
#include <opencv2/core.hpp>
#include <thread>


THIRD_PARTY_HEADERS_END



#include "common/node_creation_makros.h"

SensorCameraNode::SensorCameraNode(ros::NodeHandle &node_handle)
    : NodeBase(node_handle), sensor_camera_(&parameter_handler_) {

  // TODO: Load Cropping region!

  // Using the same input/output queue size as perception / preprocessing!
  // parameter_handler_.registerParam(INPUT_QUEUE_SIZE);
  // parameter_handler_.registerParam(OUTPUT_QUEUE_SIZE);
}

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

CREATE_NODE(SensorCameraNode)
