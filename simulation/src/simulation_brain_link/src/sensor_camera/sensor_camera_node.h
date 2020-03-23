#ifndef SENSOR_CAMERA_NODE_H
#define SENSOR_CAMERA_NODE_H
#include <common/macros.h>

THIRD_PARTY_HEADERS_BEGIN
#include <image_transport/image_transport.h>
THIRD_PARTY_HEADERS_END

#include "common/node_base.h"

#include "sensor_camera.h"
/*!
 * \brief Precrops camera image outputted by Gazebo
 */
class SensorCameraNode : public NodeBase {
 public:
  /*!
   * \brief SensorCameraNode the constructor.
   * \param node_handle the NodeHandle to be used.
   */
  SensorCameraNode(ros::NodeHandle& node_handle);
  /*!
   * \brief returns the name of the node. It is needed in main and onInit
   * (nodelet) method.
   * \return the name of the node
   */
  static std::string getName();

 private:
  // TODO: Check what these parameters are necessary for and if they should be
  // used in this context!
  static const ParameterString<int> INPUT_QUEUE_SIZE;
  static const ParameterString<int> OUTPUT_QUEUE_SIZE;

  // NodeBase interface
  /*!
   * \brief startModule is called, if the node shall be turned active. In here
   * the subrscribers an publishers are started.
   */
  virtual void startModule() override;
  /*!
   * \brief stopModule is called, if the node shall be turned inactive. In this
   * function subscribers and publishers are shutdown.
   */
  virtual void stopModule() override;


  /**
    * @brief handleImage
    *
    * Call-back for incomming images.
    * Receives image from Gazebo Camera; publishes precropped image.
    *
    * @param msg the message
    */
  void handleImage(const sensor_msgs::ImageConstPtr& msg);


  /**
     * @brief rossub_uncropped_image
     *
     * Subscribes camera image.
     */
  image_transport::Subscriber rossub_uncropped_image;
  /**
   * @brief rospub_image
   *
   * Publisher for precropped images.
   */
  ros::Publisher rospub_image;



  /*!
   * \brief sensor_camera contains the ROS-indipendent implementation of this
   * node.
   */
  SensorCamera sensor_camera_;
};

#endif  // PRE_CROPPING_NODE_H
