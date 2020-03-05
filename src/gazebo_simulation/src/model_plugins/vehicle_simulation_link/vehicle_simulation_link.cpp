#include "vehicle_simulation_link.h"

THIRD_PARTY_HEADERS_BEGIN
#include <ros/ros.h>
#include <std_msgs/Empty.h>
#include <tf2/LinearMath/Quaternion.h>
#include <tf2/convert.h>
#include <tf2_ros/transform_broadcaster.h>
#include <tf2_ros/transform_listener.h>
#include <eigen3/Eigen/Geometry>
#include <ignition/math/Pose3.hh>

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

  if (node_handle_.param("enable_z_coord", true)) {
    update_connection = event::Events::ConnectWorldUpdateBegin(
        std::bind(&VehicleSimulationLink::onWorldUpdate, this));
  } else {
    update_connection = event::Events::ConnectWorldUpdateBegin(
        std::bind(&VehicleSimulationLink::onWorldUpdateWithoutZ, this));
  }
  pose_param = node_handle_.param("set_pose", false);
  twist_param = node_handle_.param("set_twist", true);

  activateIfDesired();
}

void VehicleSimulationLink::onWorldUpdate() {

  if (set_pose && pose_param) {
    model->SetWorldPose(pose, true, false);
    set_pose = false;
  }

  if (twist_param) {
    const auto pose = model->WorldPose();
    auto rot = pose.Rot();
    Eigen::Quaternion<double> rotation(rot.W(), rot.X(), rot.Y(), rot.Z());
    Eigen::Quaternion<double> vehicle_to_world = rotation.normalized();
    Eigen::Vector3d velocity_in_world = vehicle_to_world * speed;
    ignition::math::Vector3 prev_velocity_in_world = model->WorldLinearVel();
    model->SetLinearVel(ignition::math::Vector3d(
        velocity_in_world.x(), velocity_in_world.y(), prev_velocity_in_world.Z()));
    ignition::math::Vector3 prev_angular_velocity_in_world = model->WorldAngularVel();
    model->SetAngularVel(ignition::math::Vector3d(
        prev_angular_velocity_in_world.X(), prev_angular_velocity_in_world.Y(), yaw_rate));
  }
}

void VehicleSimulationLink::onWorldUpdateWithoutZ() {

  if (set_pose && pose_param) {
    model->SetWorldPose(pose, true, false);
    set_pose = false;
  }

  if (twist_param) {
    const auto pose = model->WorldPose();
    auto rot = pose.Rot();
    Eigen::Quaternion<double> rotation(rot.W(), rot.X(), rot.Y(), rot.Z());
    Eigen::Quaternion<double> vehicle_to_world = rotation.normalized();
    Eigen::Vector3d velocity_in_world = vehicle_to_world * speed;
    model->SetLinearVel(ignition::math::Vector3d(
        velocity_in_world.x(), velocity_in_world.y(), 0));
    model->SetAngularVel(ignition::math::Vector3d(0, 0, yaw_rate));
  }
}

void VehicleSimulationLink::simulationWorldTransformation(const geometry_msgs::TransformStamped wv_stamped) {

  // Vehicle in simulation == transformation from vehicle to simulation
  ignition::math::Pose3 vs_pose = model->WorldPose();

  // To ignition pose
  geometry_msgs::Transform wv_tf = wv_stamped.transform;
  ignition::math::Pose3d wv_pose(wv_tf.translation.x,
                                 wv_tf.translation.y,
                                 wv_tf.translation.z,
                                 wv_tf.rotation.w,
                                 wv_tf.rotation.x,
                                 wv_tf.rotation.y,
                                 wv_tf.rotation.z);
  ignition::math::Pose3d vw_pose = wv_pose.Inverse();

  // Simulation to world == sim to vehicle -> vehicle to world
  // This transformation works (Confused why tho...): it seems like vehicle to
  // world + vehicle to sim is the right transformation
  ignition::math::Pose3d sw_pose = vw_pose + vs_pose;

  // Publish Transformation

  static tf2_ros::TransformBroadcaster br;
  geometry_msgs::TransformStamped transformStamped;

  transformStamped.header.stamp = ros::Time::now();
  transformStamped.header.frame_id = "simulation";
  transformStamped.child_frame_id = "world";
  transformStamped.transform.translation.x = sw_pose.Pos().X();
  transformStamped.transform.translation.y = sw_pose.Pos().Y();
  transformStamped.transform.translation.z = sw_pose.Pos().Z();

  transformStamped.transform.rotation.w = sw_pose.Rot().W();
  transformStamped.transform.rotation.x = sw_pose.Rot().X();
  transformStamped.transform.rotation.y = sw_pose.Rot().Y();
  transformStamped.transform.rotation.z = sw_pose.Rot().Z();

  br.sendTransform(transformStamped);
}

void VehicleSimulationLink::stateCallback(const state_estimation_msgs::StateConstPtr& msg) {
  speed = Eigen::Vector3d{msg->speed_x, msg->speed_y, 0};
  yaw_rate = msg->yaw_rate;
}

void VehicleSimulationLink::tfCallback(const tf2_msgs::TFMessageConstPtr& msgs) {
  for (const auto& msg : msgs->transforms) {
    if (msg.header.frame_id == "world" && msg.child_frame_id == "vehicle") {
      simulationWorldTransformation(msg);
      pose = ignition::math::Pose3d(
          ignition::math::Vector3d(msg.transform.translation.x,
                                   msg.transform.translation.y,
                                   msg.transform.translation.z),
          ignition::math::Quaterniond(msg.transform.rotation.w,
                                      msg.transform.rotation.x,
                                      msg.transform.rotation.y,
                                      msg.transform.rotation.z));
      set_pose = true;
    }
  }
}

void VehicleSimulationLink::resetCallback(__attribute__((unused))
                                          const std_msgs::Header& msg) {
  speed = Eigen::Vector3d::Zero();
  yaw_rate = 0.0;

  model->SetWorldPose(ignition::math::Pose3d(0, -0.2, 0, 0, 0, 0));
  model->SetLinearVel(ignition::math::Vector3d(0, 0, 0));
  model->SetAngularVel(ignition::math::Vector3d(0, 0, 0));
}


void VehicleSimulationLink::startModule() {
  state_subscriber = node_handle_.subscribe(
      "/state_estimation/state_estimation/state_estimation",
      1,
      &VehicleSimulationLink::stateCallback,
      this);
  tf_subscriber =
      node_handle_.subscribe("/tf", 8, &VehicleSimulationLink::tfCallback, this);

  reset_subscriber = node_handle_.subscribe(
      "/simulation/reset_location", 1, &VehicleSimulationLink::resetCallback, this);

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
