#include "model_plugin_link.h"

namespace gazebo {
// Register this plugin with the simulator
GZ_REGISTER_MODEL_PLUGIN(ModelPluginLink)

ModelPluginLink::ModelPluginLink()
    : node_handle_(ros::NodeHandle("~model_plugin_link")) {}

// Load the controller
void ModelPluginLink::Load(physics::ModelPtr _parent, sdf::ElementPtr /*_sdf*/) {
  // Make sure the ROS node for Gazebo has already been initalized
  if (!ros::isInitialized()) {
    ROS_FATAL_STREAM_NAMED("model_plugin_link",
                           "A ROS node for Gazebo has not been initialized, "
                           "unable to load plugin. "
                               << "Load the Gazebo system plugin "
                                  "'libgazebo_ros_api_plugin.so' in the "
                                  "gazebo_ros package)");
    return;
  }

  model = _parent;

  // Connects the Gazebo update cycle with the onWorldUpdate function.
  update_connection = event::Events::ConnectWorldUpdateBegin(
      std::bind(&ModelPluginLink::onWorldUpdate, this));

  std::string model_name = model->GetName();
  std::string topic = "/simulation/gazebo/model/" + model_name;

  ROS_INFO_STREAM("Advertising model link services for "
                  << model_name << " in the namespace " << topic);

  // twist_publisher =
  //  node_handle_.advertise<geometry_msgs::Twist>(topic + "/twist", 1, true);
  twist_subscriber = node_handle_.subscribe(
      topic + "/set_twist", 1, &ModelPluginLink::setTwistCallback, this);

  pose_subscriber = node_handle_.subscribe(
      topic + "/set_pose", 1, &ModelPluginLink::setPoseCallback, this);

  pose_publisher = node_handle_.advertise<geometry_msgs::Pose>(topic + "/pose", 1, true);
  twist_publisher =
      node_handle_.advertise<geometry_msgs::Twist>(topic + "/twist", 1, true);
}

void ModelPluginLink::onWorldUpdate() {

  // Check if there's a new twist to be set
  if (set_pose) {
    auto pose = ignition::math::Pose3d(desired_pose[0],
                                       desired_pose[1],
                                       desired_pose[2],
                                       desired_pose[3],
                                       desired_pose[4],
                                       desired_pose[5],
                                       desired_pose[6]);
    model->SetWorldPose(pose);
    set_pose = false;
  }

  // Check if there's a new twist to be set
  if (set_twist) {
    auto lin = ignition::math::Vector3d(
        desired_twist[0], desired_twist[1], desired_twist[2]);
    model->SetLinearVel(lin);
    auto ang = ignition::math::Vector3d(
        desired_twist[3], desired_twist[4], desired_twist[5]);
    model->SetAngularVel(ang);
    set_twist = false;
  }
  publishPose();
  publishTwist();
}


void ModelPluginLink::setTwistCallback(const gazebo_simulation::SetModelTwist &msg) {
  // Get current twist
  const auto lin = model->WorldLinearVel();
  const auto ang = model->WorldAngularVel();
  desired_twist =
      std::vector<double>{lin.X(), lin.Y(), lin.Z(), ang.X(), ang.Y(), ang.Z()};

  // List of all new values
  for (unsigned int i = 0; i < std::min(msg.keys.size(), msg.values.size()); i++) {
    desired_twist[msg.keys[i]] = msg.values[i];
  }
  set_twist = true;
}

void ModelPluginLink::setPoseCallback(const gazebo_simulation::SetModelPose &msg) {

  // Get current pose
  const auto pose = model->WorldPose();
  const auto pos = pose.Pos();
  const auto rot = pose.Rot();
  desired_pose = std::vector<double>{
      pos.X(), pos.Y(), pos.Z(), rot.W(), rot.X(), rot.Y(), rot.Z()};


  for (unsigned int i = 0; i < std::min(msg.keys.size(), msg.values.size()); i++) {
    desired_pose[msg.keys[i]] = msg.values[i];
  }

  set_pose = true;
}


void ModelPluginLink::publishTwist() {

  // Get current twist
  const auto lin = model->WorldLinearVel();
  const auto ang = model->WorldAngularVel();

  auto new_twist = geometry_msgs::Twist();
  new_twist.linear.x = lin.X();
  new_twist.linear.y = lin.Y();
  new_twist.linear.z = lin.Z();
  new_twist.angular.x = ang.X();
  new_twist.angular.y = ang.Y();
  new_twist.angular.z = ang.Z();

  twist_publisher.publish(new_twist);
}

void ModelPluginLink::publishPose() {

  // Get current pose
  const auto pose = model->WorldPose();
  const auto pos = pose.Pos();
  const auto rot = pose.Rot();

  auto new_pose = geometry_msgs::Pose();
  new_pose.position.x = pos.X();
  new_pose.position.y = pos.Y();
  new_pose.position.z = pos.Z();
  new_pose.orientation.w = rot.W();
  new_pose.orientation.x = rot.X();
  new_pose.orientation.y = rot.Y();
  new_pose.orientation.z = rot.Z();

  pose_publisher.publish(new_pose);
}

}  // namespace gazebo
