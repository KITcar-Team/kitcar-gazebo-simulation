#include "segmentation_node.h"

THIRD_PARTY_HEADERS_BEGIN
#include <cv_bridge/cv_bridge.h>
#include <sensor_msgs/image_encodings.h>
#include <thread>

THIRD_PARTY_HEADERS_END



#include "common/node_creation_makros.h"

SegmentationNode::SegmentationNode(ros::NodeHandle &node_handle)
    : NodeBase(node_handle), segmentation_(&parameter_handler_) {

  // TODO: Load Cropping region!

  // Using the same input/output queue size as perception / preprocessing!
  // parameter_handler_.registerParam(INPUT_QUEUE_SIZE);
  // parameter_handler_.registerParam(OUTPUT_QUEUE_SIZE);
}

void SegmentationNode::startModule() {


  // const int input_queue_size = parameter_handler_.getParam(INPUT_QUEUE_SIZE);

  image_transport::ImageTransport img_trans(node_handle_);
  rossub_uncropped_image = img_trans.subscribe(
      "/camera/image_uncropped_raw", 1, &SegmentationNode::handleImage, this);
}

void SegmentationNode::stopModule() {
  rossub_uncropped_image.shutdown();
}

void SegmentationNode::handleImage(const sensor_msgs::ImageConstPtr &msg) {

  cv_bridge::CvImageConstPtr cv_ptr;
  try {
    cv_ptr = cv_bridge::toCvShare(msg, sensor_msgs::image_encodings::RGB8);
  } catch (const cv_bridge::Exception &e) {
    ROS_ERROR("cv_bridge exception: %s", e.what());
    return;
  }

  cv_bridge::CvImage out_msg;
  out_msg.header = msg->header;
  out_msg.encoding = sensor_msgs::image_encodings::RGB8;

  segmentation_.modifyImage(cv_ptr->image, out_msg.image);

  cv::Mat default_img,segmentation_img;

  cv::cvtColor(out_msg.image, default_img, CV_RGB2GRAY);

  segmentation_img = out_msg.image.clone();
  //

  segmentation_.storeImages(default_img,segmentation_img);

}


std::string SegmentationNode::getName() { return std::string("segmentation"); }

CREATE_NODE(SegmentationNode)
