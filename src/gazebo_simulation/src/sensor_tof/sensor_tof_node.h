#ifndef SENSOR_TOF_NODE_H
#define SENSOR_TOF_NODE_H
#include <common/macros.h>

THIRD_PARTY_HEADERS_BEGIN

#include <common_msgs/Float32Stamped.h>

#include <sensor_msgs/PointCloud2.h>
THIRD_PARTY_HEADERS_END

#include "common/node_base.h"

#include "sensor_tof.h"

/*!
 * \brief Used to publish time of flight sensors data in simulation
 */
class SensorTofNode : public NodeBase {
 public:
  /*!
   * \brief SensorTofNode the constructor.
   * \param node_handle the NodeHandle to be used.
   */
  SensorTofNode(ros::NodeHandle& node_handle);
  /*!
   * \brief returns the name of the node. It is needed in main and onInit
   * (nodelet) method.
   * \return the name of the node
   */
  static std::string getName();

 private:
  std::string sub_topic;
  std::string pub_topic;
  std::string frame_id;

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
    * @brief handleRawDepthImage
    *
    * Call-back for incomming depth images.
    * Receives image from Gazebo Depth Camera; publishes tof sensor output.
    *
    * @param msg the message
    */
  void handleRawDepthImage(const sensor_msgs::PointCloud2ConstPtr& msg);


  /**
     * @brief rossub_raw_depth_back
     *
     * Subscribe to depth camera images.
     */
  ros::Subscriber rossub_raw_depth;

  /**
   * @brief rospub_image
   *
   * Publisher for tof outputs.
   */
  ros::Publisher rospub_tof;



  /*!
   * \brief sensor_tof contains the ROS-indipendent implementation of
   * this node.
   */
  SensorTof sensor_tof_;
};

#endif  // SENSOR_TOF_NODE_H
