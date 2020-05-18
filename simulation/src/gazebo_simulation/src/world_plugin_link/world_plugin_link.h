#ifndef WORLD_PLUGIN_LINK_H
#define WORLD_PLUGIN_LINK_H

#include <thread>

#include "gazebo/common/common.hh"
#include "gazebo/gazebo.hh"
#include "gazebo/physics/physics.hh"
#include "ros/ros.h"
#include "std_msgs/String.h"

class WorldPluginLink : public gazebo::WorldPlugin {
 public:
  /// \brief Constructor
  WorldPluginLink();

  /// \brief Destructor
  virtual ~WorldPluginLink() = default;

  /// \brief Load the controller
  virtual void Load(gazebo::physics::WorldPtr _parent, sdf::ElementPtr /*_sdf*/) override;

 private:
  //! \brief Spawn a new model in Gazebo.
  //! \param msg Sdf model definition as a string.
  void spawnModelCallback(const std_msgs::String &msg);

  //! \brief Remove a model from Gazebo.
  //! \param msg Name of the model.
  void removeModelCallback(const std_msgs::String &msg);

  //! Subscriber for spawning new model
  ros::Subscriber spawn_subscriber;
  //! Subscriber for removing a model
  ros::Subscriber remove_subscriber;

  //! Gazebo world pointer
  gazebo::physics::WorldPtr world;

  //! Handle to communicate with the ROS node
  ros::NodeHandle node_handle_;
};

#endif  // WORLD_PLUGIN_LINK_H
