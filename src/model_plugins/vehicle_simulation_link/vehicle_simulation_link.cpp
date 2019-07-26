#include "vehicle_simulation_link.h"

THIRD_PARTY_HEADERS_BEGIN
#include <ros/ros.h>
#include <std_msgs/Empty.h>
#include <tf2/convert.h>
#include <eigen3/Eigen/Geometry>
THIRD_PARTY_HEADERS_END

namespace gazebo {
// Register this plugin with the simulator
GZ_REGISTER_MODEL_PLUGIN(VehicleSimulationLink)

VehicleSimulationLink::VehicleSimulationLink()
    : NodeBase(ros::NodeHandle("~vehicle_simulation_link")),
      speed(Eigen::Vector3d::Zero()) {}

// Load the controller
void VehicleSimulationLink::Load(physics::ModelPtr _parent, sdf::ElementPtr /*_sdf*/) {
  // Make sure the ROS node for Gazebo has already been initalized
  if (!ros::isInitialized()) {
    ROS_FATAL_STREAM_NAMED("vehicle_simulation_link",
                           "A ROS node for Gazebo has not been initialized, "
                           "unable to load plugin. "
                               << "Load the Gazebo system plugin "
                                  "'libgazebo_ros_api_plugin.so' in the "
                                  "gazebo_ros package)");
    return;
  }


  model = _parent;
  update_connection = event::Events::ConnectWorldUpdateBegin(
      std::bind(&VehicleSimulationLink::onWorldUpdate, this));

  activateIfDesired();
}

void VehicleSimulationLink::onWorldUpdate() {
  if (set_pose && node_handle_.param<bool>("set_pose", false)) {
    model->SetWorldPose(pose, true, false);
    set_pose = false;
  }

  if (node_handle_.param<bool>("set_twist", true)) {
    const auto pose = model->WorldPose();
    Eigen::Quaternion<double> rotation(
        pose.Rot().W(), pose.Rot().X(), pose.Rot().Y(), pose.Rot().Z());
    Eigen::Quaternion<double> vehicle_to_world = rotation.normalized();
    Eigen::Vector3d velocity_in_world = vehicle_to_world * speed;
    Eigen::Vector3d angular_velocity_in_world =
        vehicle_to_world * Eigen::Vector3d(0.0, 0.0, yaw_rate);
    model->SetLinearVel(ignition::math::Vector3d(
        velocity_in_world.x(), velocity_in_world.y(), velocity_in_world.z()));
    model->SetAngularVel(ignition::math::Vector3d(angular_velocity_in_world.x(),
                                                  angular_velocity_in_world.y(),
                                                  angular_velocity_in_world.z()));
  }
}

void VehicleSimulationLink::stateCallback(const state_estimation_msgs::StateConstPtr& msg) {
  speed = Eigen::Vector3d{msg->speed_x, msg->speed_y, 0};
  yaw_rate = msg->yaw_rate;
}

void VehicleSimulationLink::tfCallback(const tf2_msgs::TFMessageConstPtr& msgs) {
  for (const auto& msg : msgs->transforms) {
    if (msg.header.frame_id == "world" && msg.child_frame_id == "vehicle") {
      /*pose = ignition::math::Pose3d(
          ignition::math::Vector3d(msg.transform.translation.x,
                                   msg.transform.translation.y,
                                   msg.transform.translation.z),
          ignition::math::Quaterniond(msg.transform.rotation.w,
                                      msg.transform.rotation.x,
                                      msg.transform.rotation.y,
                                      msg.transform.rotation.z));
      set_pose = true;*/
    }
  }
}

void VehicleSimulationLink::startModule() {
  state_subscriber = node_handle_.subscribe(
      "/state_estimation/state_estimation/state_estimation",
      1,
      &VehicleSimulationLink::stateCallback,
      this);
  tf_subscriber =
      node_handle_.subscribe("/tf", 8, &VehicleSimulationLink::tfCallback, this);

  const auto pose = model->WorldPose();
  ros::param::set("/navigation/localization/initial_x", pose.Pos().X());
  ros::param::set("/navigation/localization/initial_y", pose.Pos().Y());
  ros::param::set("/navigation/localization/initial_yaw", pose.Rot().Yaw());

  reset_localization = node_handle_.advertise<std_msgs::Empty>(
      "/navigation/localization/reset_location", 1, true);
  reset_localization.publish(std_msgs::Empty());
}

void VehicleSimulationLink::stopModule() {
  state_subscriber.shutdown();
  speed = Eigen::Vector3d::Zero();
  yaw_rate = 0.0;

  reset_localization.shutdown();
}
}  // namespace gazebo