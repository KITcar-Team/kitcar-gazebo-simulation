
import rospy

from common_msgs.msg import MissionMode
from std_msgs.msg import Float32,Int8

import pdb

from enum import Enum

class DriveState(Enum):
    OFF = 0
    IN_PROGRESS = 1
    COMPLETED = 2
    FAILED = 3


class SimulationDriveLoggerNode:
    """ Complete an automatic test using this node. 
        Also the gazebo simulation, KITcar_brain and the CarStateNode needs to be running

    """
    def __init__(self):
        rospy.init_node('simulation_drive_logger')


        #Read required parameters        
        self.car_name = rospy.get_param('~car_name')

        #Read optional parameters
        self.topic_env = rospy.get_param('~topic_environment','/simulation/drive_logger/')
        self.start_activated = rospy.get_param('~start_activated', True)
        self.tolerance = float(rospy.get_param('~tolerance',0.1)) #Car can leave the corridor for this amount of seconds

        print(f'Tolerance:{self.tolerance}') 

        self.mission_mode_publisher = rospy.Publisher('/mission_mode',MissionMode,queue_size=100) 
        
        self.state_publisher = rospy.Publisher(self.topic_env + 'state',Int8,queue_size=1)

        rospy.wait_for_message('/simulation/car_state/progress',Float32)

        if self.start_activated:
            self.start()
        rospy.spin()  

        self.stop()

    def start(self):
        self.progress_subscriber = rospy.Subscriber(f"/simulation/car_state/progress",Float32, callback=self.progress_cb)
        self.start_test()
        pass

    def stop(self):
        self.mission_mode_publisher.unregister()
        self.state_publisher.unregister()
        self.progress_subscriber.unregister()
        pass

    def start_test(self):
        self.last_on_track = rospy.Time()
        self.change_mission_mode(MissionMode.FREE_RIDE)
        self.start_time = rospy.Time()
        self.state = DriveState.IN_PROGRESS
        pass


    def change_mission_mode(self, mode = MissionMode.FREE_RIDE):

        if not mode is MissionMode:
            mode = MissionMode.FREE_RIDE

        msg = MissionMode()
        msg.header.stamp = rospy.get_rostime()
        msg.mission_mode = mode
        print(f'change_mission_mode to {mode}') 
        self.mission_mode_publisher.publish(msg)
    

    def progress_cb(self, msg):
        progress = msg.data
        self.update_state(progress)
        self.publish_state()
 
    def update_state(self, progress):
        print(progress)

        if self.state != DriveState.IN_PROGRESS:
            return

        if progress == 1:
            self.completion_time = rospy.Time()
            self.state = DriveState.COMPLETED
            
            return
        print( rospy.Time.now().to_sec(), self.last_on_track.to_sec()) 
        if progress != -1:
            self.last_on_track = rospy.Time.now()
        elif rospy.Time().now().to_sec() - self.last_on_track.to_sec() > self.tolerance:
            self.state = DriveState.FAILED
            return

    def publish_state(self):
        msg = Int8()
        msg.data = self.state.value

        self.state_publisher.publish(msg)


            

