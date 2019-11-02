#ifndef VEHICLE_SIMULATION_LINK_H
#define VEHICLE_SIMULATION_LINK_H

#include <common/macros.h>

#include "common/node_base.h"


THIRD_PARTY_HEADERS_BEGIN
#include <ros/ros.h>

#include <eigen3/Eigen/Core>

#include <gazebo/common/Events.hh>
#include <gazebo/common/Plugin.hh>
#include <gazebo/common/Time.hh>
#include <gazebo/physics/physics.hh>
#include <gazebo/transport/TransportTypes.hh>

#include <tf2_msgs/TFMessage.h>
#include "state_estimation_msgs/State.h"
THIRD_PARTY_HEADERS_END


namespace gazebo {

class VehicleSimulationLink : public ModelPlugin, public NodeBase {
 public:
  /// \brief Constructor
  VehicleSimulationLink();

  /// \brief Destructor
  virtual ~VehicleSimulationLink() = default;

  /// \brief Load the controller
  virtual void Load(physics::ModelPtr _parent, sdf::ElementPtr /*_sdf*/) override;


 private:
  //!
  //! \brief updates the model's linear and angular velocity which makes the robot move
  //! \todo How frequent is this function called? Are we dropping state estimation messages?
  //!
  void onWorldUpdate();

  //!
  //! \brief updates the
  //! \param msg incoming state estimate
  //!
  void stateCallback(const state_estimation_msgs::StateConstPtr &msg);
	

	void simulationWorldTransformation(const geometry_msgs::TransformStamped wv_stamped);

  //!
  //! \brief obtains the robot's pose from /tf
  //! \param msgs tf message
  //!
  void tfCallback(const tf2_msgs::TFMessageConstPtr &msgs);

  //!
  //! \brief startModule initializes publishers and subsribers and sets the
  //! /navigation/localization/initial... parameters by robot's initial pose
  //!
  virtual void startModule() override;

  //!
  //! \brief stopModule stops the publishers and subsribers
  //!
  virtual void stopModule() override;

  ros::Subscriber state_subscriber;
  ros::Subscriber tf_subscriber;

  ros::Publisher reset_localization;  //! publisher to trigger a navigation localization reset

  Eigen::Vector3d speed;  //! the robots velocity obtained by the state estimation
  double yaw_rate = 0.0;  //! the robots yaw rate obtained by the state estimation

  ignition::math::Pose3d pose;  //! the robots pose obtained by /tf (from state estimation) to prevent a drift
  bool set_pose = false;  //! flag indicating that the robot's pose (\see model) should be updated (new pose received)

  physics::ModelPtr model;  //! gazebo's representation of the robot

  event::ConnectionPtr update_connection;  //! pointer to the update event connection
};
}  // namespace gazebo

#endif  // VEHICLE_SIMULATION_LINK_H
