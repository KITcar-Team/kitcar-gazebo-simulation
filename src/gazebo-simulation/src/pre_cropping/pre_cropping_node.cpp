#include "pre_cropping_node.h"

THIRD_PARTY_HEADERS_BEGIN
#include <cv_bridge/cv_bridge.h>
#include <sensor_msgs/image_encodings.h>
#include <thread>


THIRD_PARTY_HEADERS_END



#include "common/node_creation_makros.h"

PreCroppingNode::PreCroppingNode(ros::NodeHandle &node_handle)
    : NodeBase(node_handle), pre_cropping_(&parameter_handler_) {

  // TODO: Load Cropping region!

  // Using the same input/output queue size as perception / preprocessing!
  // parameter_handler_.registerParam(INPUT_QUEUE_SIZE);
  // parameter_handler_.registerParam(OUTPUT_QUEUE_SIZE);
}

void PreCroppingNode::startModule() {

  ROS_INFO("Starting module PreCroppingNode");

  // const int output_queue_size =
  // parameter_handler_.getParam(OUTPUT_QUEUE_SIZE);
  rospub_image = node_handle_.advertise<sensor_msgs::Image>("/camera/image_raw", 1);

  // const int input_queue_size = parameter_handler_.getParam(INPUT_QUEUE_SIZE);

  image_transport::ImageTransport img_trans(node_handle_);
  rossub_uncropped_image = img_trans.subscribe(
      "/camera/image_uncropped_raw", 1, &PreCroppingNode::handleImage, this);
}

void PreCroppingNode::stopModule() {
  rossub_uncropped_image.shutdown();
  rospub_image.shutdown();
}

void PreCroppingNode::handleImage(const sensor_msgs::ImageConstPtr &msg) {

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

  pre_cropping_.precropImage(cv_ptr->image, out_msg.image);

  sensor_msgs::ImagePtr out_ptr(out_msg.toImageMsg());
  rospub_image.publish(out_ptr);
}


std::string PreCroppingNode::getName() { return std::string("pre_cropping"); }

CREATE_NODE(PreCroppingNode)
