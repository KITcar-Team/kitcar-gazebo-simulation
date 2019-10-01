#!/usr/bin/env python3
# -*- coding: utf-8 -*-
""" Contains CarStateNode, CarState and CarStateUpdater """

import rospy
import os

from geometry_msgs.msg import Quaternion,Point,Pose, Vector3, Twist
from nav_msgs.msg import Odometry
from std_msgs.msg import String, Float32
from gazebo_msgs.msg import ModelStates
from visualization_msgs.msg import Marker
from geometry_msgs.msg import PointStamped

import shapely.geometry as geom
import shapely.affinity as affinity
import shapely.ops as ops

import pyquaternion

from simulation_groundtruth.groundtruth import SimulationGroundtruth


class CarStateNode:
    """ Monitors and publishes the cars state

    @car_name:String
    
    @car_state:CarState contains and calculates groundtruth and other information

    @car_state_updater:CarStateUpdater used to publish car state information

    """

    def __init__(self):
        #initialize the node
        rospy.init_node('car_state_node')

        #Read required parameters
        road = rospy.get_param('~road')
        self.car_name = rospy.get_param('~car_name')

        #Read optional parameters
        self.topic_env = rospy.get_param('~topic_environment','/simulation/car_state/')
        self.start_activated = rospy.get_param('~start_activated', True)

        road_file = os.path.join(os.environ.get('KITCAR_REPO_PATH'),'kitcar-gazebo-simulation','models','env_db',road,'road.xml')
        groundtruth = SimulationGroundtruth(road_file) 
        self.car_state = CarState(groundtruth)

        #initialize publisher
        self.car_state_updater = CarStateUpdater(self.topic_env + 'frame',self.topic_env + 'road_section',self.topic_env + 'progress')

        if self.start_activated:
            self.start()
        
        rospy.spin()
        
        self.stop()

    def start(self):
        """Start node."""
        self.car_state_updater.start()
        #Start subscribing to changes in gazebo
        self.subscriber = rospy.Subscriber('/gazebo/model_states', ModelStates, callback=self.car_state_cb, queue_size=1)
        pass

    def stop(self):
        """Turn off publisher."""
        self.subscriber.unregister()
        self.car_state_updater.stop()
        pass

    def car_state_cb(self,model_states):
        """ Called when the model_states in gazebo are updated. Updates all car_state topics.

        @model_states:ModelStates 
        """

        #Find information for current car in model_states and give to self.car_state
        idx = model_states.name.index(self.car_name)
        pose = model_states.pose[idx]
        twist = model_states.twist[idx]
        self.car_state.receive_update(pose,twist)
        
        self.car_state_updater.publish_car_frame(self.car_state.current_frame)

        #Current polygon
        poly = self.car_state.current_polygon
        self.car_state_updater.publish_current_polygon(poly)

        #Publish progress
        progress = -1 if poly is None else self.car_state.current_progress(poly)
        self.car_state_updater.publish_current_progress(progress)


class CarState:
    """
    @groundtruth: 
    """

    def __init__(self, groundtruth, car_frame = geom.Polygon([(-0.164,0.089),(0.164,0.089),(0.164,-0.089),(-0.164,-0.089)])):
        """"""
        self.groundtruth = groundtruth
        self.car_frame = car_frame
        self.current_pose = Pose(position=Point(x=0,y=0,z=0),orientation=Quaternion(x=0,y=0,z=0,w=0))
        self.current_twist = Twist(linear=Vector3(0,0,0),angular=(0,0,0))

        self.failures = 0
        self.continous_failures = 0


    def receive_update(self, pose, twist):

        self.current_pose = pose
        self.current_twist = twist    

        self.current_frame = self.transform_car_frame(pose)
        self.current_polygon = self.get_current_polygon(self.groundtruth.corridors)
        
    def transform_car_frame(self, car_pose):
        """
        Calculate position of the cars frame points as a polygon.
        """
        #Use pyquaternion to retrieve angle between car and simulation axes
        angle = pyquaternion.Quaternion(car_pose.orientation.w, car_pose.orientation.x, car_pose.orientation.y, car_pose.orientation.z).degrees 
        

        rotated = affinity.rotate(self.car_frame, -angle) 
        transformed = affinity.translate(rotated,car_pose.position.x,car_pose.position.y)

        return transformed 


    def get_current_polygon(self, polygons):

        """
        Determine in which polygon the car currently is (if in any)

        @polygons: List of polygons:[ np.array of shape (n,2)] which are the points of the polygons 

        @car_pose: Pose of car as ..., pose is treated as referring to center of car_frame

        @car_frame: Frame of the car

        Return: index of the polygon that car is **completely** inside of, None otherwise
        """

        car_points = [geom.Point(coord) for coord in self.current_frame.exterior.coords]

        current_polygon = None
        point_count = 0
        for polygon in polygons: 
            new_points = 0
            for p in car_points:#Last point is also first point!
                if p.within(polygon):
                    new_points += 1
            if new_points > 0:
                point_count += new_points
                current_polygon = polygon
                #print('True')

            if point_count >= 4:
                break


        #For debugging 
        if point_count < 4:
            self.failures += 1
            self.continous_failures += 1
            print(f"{self.failures} in total, {self.continous_failures} in a row")
        else:
            self.continous_failures = 0

        return current_polygon if point_count >=4 else None


    def current_progress(self, polygon):

        try:
            idx = self.groundtruth.corridors.index(polygon)
        except Exception as e:
            print("Failed to find current polygon in list of polygons: ")
            print(e)
            return -1
        else:
            return idx / (len(self.groundtruth.corridors)-1)

class CarStateUpdater:
    def __init__(self, car_frame_topic, polygon_topic,progress_topic, queue_size = 1):
        
        self.frame_publisher = rospy.Publisher(car_frame_topic, Marker, queue_size=queue_size)
        self.polygon_publisher = rospy.Publisher(polygon_topic,Marker,queue_size=queue_size)
        self.progress_publisher = rospy.Publisher(progress_topic, Float32,queue_size=queue_size)

    def start(self):
        pass

    def stop(self):
        self.frame_publisher.unregister()
        self.polygon_publisher.unregister()
        self.progress_publisher.unregister()
        pass
        
    def publish_car_frame(self, current_frame):
        """ 
        Publish the polygon, that is given as the cars current frame.
        """
        marker = Marker()
        marker.header.frame_id = "simulation"
        marker.header.stamp = rospy.get_rostime()
        marker.ns = "car_state/frame"
        marker.lifetime = rospy.Duration(secs = 10)
        marker.color.r = 1
        marker.color.g = 0
        marker.color.b = 0
        marker.color.a = 1
        marker.scale.x = 0.02
        marker.scale.y = 1
        marker.scale.z = 1
        marker.id = 0

        marker.type = 4 #Polygon 
        #print(f'Car frame:') 
        for point in current_frame.exterior.coords:
            state = PointStamped()
            state.point.x = point[0]
            state.point.y = point[1]
            state.point.z = 0

            #print(point)

            marker.points.append(state.point)

        self.polygon_publisher.publish(marker)

    def publish_current_polygon(self, polygon):
        """
        Publish the polygon, that is given as the cars current polygon.
        """
        marker = Marker()
        marker.header.frame_id = "simulation"
        marker.header.stamp = rospy.get_rostime()
        marker.ns = "car_state/polygon"
        marker.lifetime = rospy.Duration(secs = 10)
        marker.color.r = 0
        marker.color.g = 0
        marker.color.b = 1
        marker.color.a = 0.8
        marker.scale.x = 0.02
        marker.scale.y = 1
        marker.scale.z = 1
        marker.id = 0

        marker.type = 4 #Polygon 
        
        if not polygon is None:

            for point in polygon.exterior.coords:
                state = PointStamped()
                state.point.x = point[0]
                state.point.y = point[1]
                state.point.z = 0

                marker.points.append(state.point)

        self.polygon_publisher.publish(marker)

    def publish_current_progress(self, progress):
        msg = Float32(data=progress)
        self.progress_publisher.publish(msg)
        
        


