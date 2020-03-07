#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""CarStateVisualizationNode"""

import rospy

from visualization_msgs.msg import Marker

from gazebo_simulation.msg import CarState as CarStateMsg

from geometry.point import Point

from ros_base.node_base import NodeBase
import ros_base.visualization as visualization

__copyright__ = "KITcar"


class CarStateVisualizationNode(NodeBase):
    """ROS node which allows to visualize the car state in rviz.

    ROS Parameters (by default they are defined in gazebo_simulation/param/car_state.yaml):
        * topics:
            * car_state(str): Subscriber topic of car_state
            * rviz:
                * cone (str): Topic to publish the view cone
                * frame (str): Topic to publish the car frame

    Attributes:
        frame_publisher (rospy.publisher): Publishes the cars frame as a rviz marker.
        view_cone_publisher (rospy.publisher): Publishes the cars view cone as a rviz marker.
        state_subscriber (rospy.subscriber): Subscribes to car_state

    """

    def __init__(self):
        """ initialize the node"""

        super(CarStateVisualizationNode, self).__init__(name="car_state_visualization_node")

        self.run()

    def start(self):
        """ Start visualization. """
        self.frame_publisher = rospy.Publisher(self.param.topics.rviz.frame, Marker, queue_size=1)
        self.view_cone_publisher = rospy.Publisher(self.param.topics.rviz.cone, Marker, queue_size=1)
        self.state_subscriber = rospy.Subscriber(self.param.topics.car_state, CarStateMsg, callback=self.state_cb)
        super().start()

    def stop(self):
        """ Stop visualization. """
        self.state_subscriber.unregister()
        self.frame_publisher.unregister()
        self.view_cone_publisher.unregister()
        super().stop()

    def state_cb(self, msg: CarStateMsg):
        """ Called when car state is published

        Arguments:
            msg (CarStateMsg): Msg published by car state node

        """
        frame_marker = visualization.get_marker_for_points(
            (Point(p) for p in msg.frame.points), frame_id="simulation", rgba=[0, 0, 1, 0.7]
        )
        self.frame_publisher.publish(frame_marker)

        if len(msg.view_cone.points):
            cone_marker = visualization.get_marker_for_points(
                (Point(p) for p in msg.view_cone.points), frame_id="simulation", id=1
            )
            self.view_cone_publisher.publish(cone_marker)
