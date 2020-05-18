#include "world_plugin_link.h"

constexpr bool DEBUG = false;


// Spawn a new model from sdf string into the world.
void spawnModel(gazebo::physics::WorldPtr world, std_msgs::String msg) {
  if (DEBUG) {
    ROS_INFO_STREAM("Spawning model " << msg.data);
  } else {
    ROS_INFO_STREAM("Spawning model");
  }

  // Insert model from string
  sdf::SDF modelSDF;
  modelSDF.SetFromString(msg.data);
  world->InsertModelSDF(modelSDF);
}

// Remove a model in the world.
void removeModel(gazebo::physics::WorldPtr world, std_msgs::String msg) {
  if (DEBUG) {
    ROS_INFO_STREAM("Removing model with name " << msg.data);
  } else {
    ROS_INFO_STREAM("Removing model");
  }

  world->RemoveModel(msg.data);
}


// Register this plugin with the simulator
GZ_REGISTER_WORLD_PLUGIN(WorldPluginLink)

WorldPluginLink::WorldPluginLink()
    : node_handle_(ros::NodeHandle("~world_plugin_link")) {}

// Load the controller
void WorldPluginLink::Load(gazebo::physics::WorldPtr _parent, sdf::ElementPtr /*_sdf*/) {
  // Make sure the ROS node for Gazebo has already been initalized
  if (!ros::isInitialized()) {
    ROS_FATAL_STREAM_NAMED("world_plugin_link",
                           "A ROS node for Gazebo has not been initialized, "
                           "unable to load plugin. "
                               << "Load the Gazebo system plugin "
                                  "'libgazebo_ros_api_plugin.so' in the "
                                  "gazebo_ros package)");
    return;
  }

  world = _parent;

  std::string topic = "/simulation/gazebo/world/";
  ROS_INFO_STREAM("Advertising world link topics in the namespace " << topic);
  spawn_subscriber = node_handle_.subscribe(
      topic + "/spawn_sdf_model", 1000, &WorldPluginLink::spawnModelCallback, this);
  remove_subscriber = node_handle_.subscribe(
      topic + "/remove_model", 1000, &WorldPluginLink::removeModelCallback, this);
}

void WorldPluginLink::spawnModelCallback(const std_msgs::String &msg) {
  // Spawn new model inside a new thread.
  std::thread task(spawnModel, world, msg);
  task.detach();
}

void WorldPluginLink::removeModelCallback(const std_msgs::String &msg) {
  // Remove a model inside a new thread.
  std::thread task(removeModel, world, msg);
  task.detach();
}
