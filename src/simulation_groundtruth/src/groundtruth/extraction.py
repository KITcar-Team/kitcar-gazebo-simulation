#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""This node extracts groundtruth information about a given road. It then makes this information available through a service / a topic."""

import rospy 

from simulation_groundtruth.msg import RoadSectionMsg

class GroundtruthExtractionNode:

    def __init__(self):

        #initialize the node
        rospy.init_node('groundtruth_extraction_node')
 
        #Read required parameters
        road_name = rospy.get_param('~road')
        self.car_name = rospy.get_param('~car_name')

        #Read optional parameters
        self.topic_env = rospy.get_param('~topic_environment','/simulation/groundtruth/')
        self.start_activated = rospy.get_param('~start_activated', True)


        #Read xml 

        #Extract groundtruth

        #Publish as RoadSectionMsg
        #A simple way to see whats inside the msg:
        #rosmsg show simulation_groundtruth.RoadSectionMsg
        

        if self.start_activated:
            self.start()

        rospy.spin()

        self.stop()

    def start(self):
        """Turn on node."""
        pass

    def stop(self):
        """Turn off."""
        pass


