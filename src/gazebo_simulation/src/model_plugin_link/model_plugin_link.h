#ifndef MODEL_PLUGIN_LINK_H
#define MODEL_PLUGIN_LINK_H

#include "gazebo_simulation/SetModelPose.h"
#include "gazebo_simulation/SetModelTwist.h"

// Ignore third party warnings!!
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wunused-variable"
#pragma GCC diagnostic ignored "-Wunknown-pragmas"
#pragma GCC diagnostic ignored "-Wunused-local-typedefs"
#pragma GCC diagnostic ignored "-Wunused-but-set-parameter"
#pragma GCC diagnostic ignored "-Wuseless-cast"
#pragma GCC diagnostic ignored "-Wignored-qualifiers"
#pragma GCC diagnostic ignored "-Wunused-parameter"
#pragma GCC diagnostic ignored "-Wempty-body"
#pragma GCC diagnostic ignored "-Wconversion"
#pragma GCC diagnostic ignored "-Wformat="
#pragma GCC diagnostic ignored "-Wsuggest-override"
#pragma GCC diagnostic ignored "-Wold-style-cast"
#pragma GCC diagnostic ignored "-Wnon-virtual-dtor"
#pragma GCC diagnostic ignored "-Wmaybe-uninitialized"
#pragma GCC diagnostic ignored "-Wzero-as-null-pointer-constant"
#pragma GCC diagnostic ignored "-Wcast-qual"
#pragma GCC diagnostic ignored "-Wlogical-op"
#pragma GCC diagnostic ignored "-Wredundant-decls"
#pragma GCC diagnostic ignored "-Wconversion"
#pragma GCC diagnostic ignored "-Wfloat-conversion"
#pragma GCC diagnostic ignored "-Wpedantic"
#pragma GCC diagnostic ignored "-Wdeprecated-declarations"

#include <ros/ros.h>
#include <eigen3/Eigen/Core>
#include <eigen3/Eigen/Geometry>

#include <geometry_msgs/Pose.h>
#include <geometry_msgs/Twist.h>
#include <std_msgs/Empty.h>
#include <gazebo/common/Events.hh>
#include <gazebo/common/Plugin.hh>
#include <gazebo/common/Time.hh>
#include <gazebo/physics/physics.hh>
#include <gazebo/transport/TransportTypes.hh>
#include <ignition/math/Pose3.hh>
#include <ignition/math/Vector3.hh>

#include <tf2/LinearMath/Quaternion.h>
#include <tf2/convert.h>
#include <tf2_msgs/TFMessage.h>
#include <tf2_ros/transform_broadcaster.h>
#include <tf2_ros/transform_listener.h>

#pragma GCC diagnostic pop

namespace gazebo {

class ModelPluginLink : public ModelPlugin {
 public:
  /// \brief Constructor
  ModelPluginLink();

  /// \brief Destructor
  virtual ~ModelPluginLink() = default;

  /// \brief Load the controller
  virtual void Load(physics::ModelPtr _parent, sdf::ElementPtr /*_sdf*/) override;

 private:
  //!
  //! \brief updates the model's linear and angular velocity which makes the robot move
  //!
  void onWorldUpdate();

  //!
  //! \brief setPoseCallback can be used to set the pose of the model in
  //! Gazebo. \param msg contains a list of values that should be updated
  //!
  void setPoseCallback(const gazebo_simulation::SetModelPose &msg);

  //!
  //! \brief setTwistCallback can be used to set the twist of the model in
  //! Gazebo. \param msg contains a list of values that should be updated
  //!
  void setTwistCallback(const gazebo_simulation::SetModelTwist &msg);

  //!
  //! \brief publishPose publishes the current pose of the model.
  //!
  void publishPose();

  //!
  //! \brief publishTwist publishes the current twist of the model.
  //!
  void publishTwist();

  bool set_twist = false;  //! signals when a new twist should be set
  bool set_pose = false;   //! signals when a new pose should be set

  ros::Publisher twist_publisher;    //! used to publish the current twist
  ros::Subscriber twist_subscriber;  //! receive requests with a new twist

  ros::Publisher pose_publisher;    //! used to publish the current pose
  ros::Subscriber pose_subscriber;  //! receive requests with a new pose

  std::vector<double> desired_twist;  //! the robots desired pose requested by calling the twist service
  std::vector<double> desired_pose;  //! the robots desired pose requested by calling the pose service

  physics::ModelPtr model;  //! gazebo's representation of the robot

  event::ConnectionPtr update_connection;  //! pointer to the update event connection

  ros::NodeHandle node_handle_;  //! handle to communicate with the ROS node
};
}  // namespace gazebo

#endif  // MODEL_PLUGIN_LINK_H
