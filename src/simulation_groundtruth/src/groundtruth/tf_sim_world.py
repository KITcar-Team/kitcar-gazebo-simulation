""" Python classes to complete simulation tf tasks. """

import tf2_ros
import geometry_msgs.msg
import rospy

from gazebo_msgs.msg import ModelStates

from pyquaternion import Quaternion 
#import tf2_geometry_msgs


class SimulationTransformNode:
    """ Monitors and publishes the simulation to world transformation
    
    @car_state:CarState contains and calculates groundtruth and other information

    @car_state_updater:CarStateUpdater used to publish car state information


    """

    def __init__(self):

        #Read parameters
        self.car_name = rospy.get_param('~car_name','dr_drift')

        #initialize the node
        rospy.init_node('sim_to_world_tf_node')
    
        self.sim_world_transformer = SimulationTransform()
        self.broadcaster = tf2_ros.TransformBroadcaster()

        #Start subscribing to changes in gazebo
        self.subscriber = rospy.Subscriber('/gazebo/model_states', ModelStates, callback=self.car_state_cb, queue_size=1)

        rospy.spin()

    def start(self):
        """Turn on publisher."""
        pass
    
    #rospy.Timer(rospy.Duration(1.0 / self.rate), self.line_cb)
 
 
    def stop(self):
        """Turn off publisher."""
        
        pass

    def car_state_cb(self,model_states):
        idx = model_states.name.index(self.car_name)
        
        pose = model_states.pose[idx]

        sim_world_tf = self.sim_world_transformer.simulation_to_world(pose)
        if not sim_world_tf is None:

            t = geometry_msgs.msg.TransformStamped()
            t.header.stamp = rospy.Time.now()
            t.header.frame_id = "simulation"
            t.child_frame_id = "world"
            t.transform.translation = sim_world_tf[0]
            t.transform.rotation = sim_world_tf[1]

            self.broadcaster.sendTransform(t)

class SimulationTransform:

    def __init__(self):
        self.buffer = tf2_ros.Buffer()
        #Start a listener
        self.listener = tf2_ros.TransformListener(self.buffer)


        pass
    
    def simulation_to_world(self, car_pose):
        
        try:
            #Get transformation from vehicle to world
            world_to_vehicle = self.buffer.lookup_transform('world','vehicle',rospy.Time())
        except Exception as e:
            print(e)
            return None

        wv_tf = world_to_vehicle.transform
        
        wv_world_translation, wv_rotation = wv_tf.translation, wv_tf.rotation 
        
        sv_sim_translation, sv_rotation = car_pose.position, car_pose.orientation
        
        wv_q = Quaternion(wv_rotation.w,wv_rotation.x,wv_rotation.y,wv_rotation.z)
        sv_q = Quaternion(sv_rotation.w,sv_rotation.x,sv_rotation.y,sv_rotation.z)
        
        

        #print(vehicle_to_world_tf, car_pose)
        #print(car_q,vw_q)
      
        q = sv_q.rotate(wv_q.inverse)
        #q = sv_q / wv_q

        rotation = geometry_msgs.msg.Quaternion()
        rotation.x = q.x
        rotation.y = q.y
        rotation.z = q.z
        rotation.w = q.w

        wv_sim_translation_list = q.inverse().rotate([wv_world_translation.x,wv_world_translation.y,wv_world_translation.z])

        translation = geometry_msgs.msg.Point()
        translation.x = sv_sim_translation.x - wv_sim_translation_list[0]       
        translation.y = sv_sim_translation.y - wv_sim_translation_list[1]       
        translation.z = sv_sim_translation.z - wv_sim_translation_list[2]

        return (translation, rotation)

