#! /usr/bin/env python3
import rospy
from simulation_groundtruth.simulation_groundtruth import SimulationGroundtruthNode


if __name__ == '__main__':
    # Initialize the node and name it.
    #rospy.init_node('simulation_groundtruth')
    # Go to class functions that do all the heavy lifting.
    try:
        SimulationGroundtruthNode()
    except rospy.ROSInterruptException:
        pass
