import xml.etree.ElementTree as ET 
from xml.etree.ElementTree import SubElement
import xml.dom.minidom as minidom
import numpy as np

#Default attributes for plugins
DEFAULT_PLUGIN_ATTRIBUTES = {'alwaysOn':1,'updateRate':0}

#Default attributes for sensors
DEFAULT_SENSOR_ATTRIBUTES = dict()

TOF_SIZE = [0.02,0.02,0.02]

def prettify(elem):
    """Return a pretty-printed XML string for the Element.
    """
    rough_string = ET.tostring(elem, 'utf-8')
    reparsed = minidom.parseString(rough_string)
    return '\n'.join([line for line in reparsed.toprettyxml(indent=' '*4).split('\n') if line.strip()])

def add_list_text(el,name, list_obj,args=dict()):
    obj = SubElement(el,name)
    obj.attrib = args
    obj.text = ' '.join(str(x) for x in list_obj)

def add_pose(el,pose):
    add_list_text(el,'pose',pose,{'frame':''})


def add_box(el,pose,size):
    if not pose is None:
        add_pose(el,pose)
    geom = SubElement(el,'geometry')
    box = SubElement(geom,'box')
    add_list_text(box,'size',size)

def add_tag_dict(el, d):
    for key in d:
        SubElement(el,key).text = str(d[key])


def create_camera_sensor(el,camera_type, plugin_type, plugin_att, horizontal_fov, image_att, clip_att, attributes):
    sensor = SubElement(el,'sensor',{'name':'camera','type':camera_type})
    
    #Default sensor attributes
    default_sensor_attrib = DEFAULT_SENSOR_ATTRIBUTES.copy()
    for key in attributes:
        default_sensor_attrib[key] = attributes[key]

    add_tag_dict(sensor,default_sensor_attrib)
    
    camera = SubElement(sensor,'camera',{'name':'__default__'})
    SubElement(camera,'horizontal_fov').text = str(horizontal_fov)
    image = SubElement(camera,'image')
    add_tag_dict(image,image_att)
    clip = SubElement(camera,'clip')
    add_tag_dict(clip,clip_att)

    plugin = SubElement(sensor,'plugin',{'name':'camera_plugin', 'filename':plugin_type})

    #Default plugin attributes
    default_plugin_attrib = DEFAULT_PLUGIN_ATTRIBUTES.copy()
    for key in plugin_att:
        default_plugin_attrib[key] = plugin_att[key]

    add_tag_dict(plugin,default_plugin_attrib)
    

def create_front_camera(el,pose,size, horizontal_fov, capture,clip, ros, attributes):
    camera = SubElement(el,'link',{'name':'camera_ros::link'})
    add_pose(camera,pose)

    #Create visual box
    visual = SubElement(camera,'visual',{'name':'visual'})
    add_box(visual,pose = None,size = size)

    #Create camera sensor 
    create_camera_sensor(camera,camera_type='camera',plugin_type='libgazebo_ros_camera.so',
        plugin_att=ros,horizontal_fov=horizontal_fov,
        image_att=capture, clip_att=clip,attributes=attributes
        )
    

    joint = SubElement(el,'joint',{'name':'camera_joint','type':'fixed'})
    add_tag_dict(joint,{'child':'camera_ros::link','parent':'chassis'})


def create_depth_camera(el,pose,size, horizontal_fov, capture,clip, ros, attributes):
    camera = SubElement(el,'link',{'name':'depth_camera_ros::link'})
    add_pose(camera,pose)

    #Create visual box
    visual = SubElement(camera,'visual',{'name':'visual'})
    add_box(visual,pose = None,size = size)

    #Create camera sensor 
    create_camera_sensor(camera,camera_type='depth',plugin_type='libgazebo_ros_openni_kinect.so',
        plugin_att=ros,horizontal_fov=horizontal_fov,
        image_att=capture, clip_att=clip,attributes=attributes
        )
    

    joint = SubElement(el,'joint',{'name':'depth_camera_joint','type':'fixed'})
    add_tag_dict(joint,{'child':'depth_camera_ros::link','parent':'chassis'})


def create_tof_camera(el, name, pose, horizontal_fov, capture,clip, topic_base, topic_info_base):
    tof = SubElement(el, 'link',{'name':'depth_camera_ros::link_' + name})
    add_pose(tof,pose)

    #Create visual box
    visual = SubElement(tof,'visual',{'name':'visual'})
    add_box(visual,pose = None,size = TOF_SIZE)

    plugin_dict = dict()
    plugin_dict['cameraName'] = 'tof_' + name 
    plugin_dict['depthImageTopicName'] = topic_base + name
    plugin_dict['depthImageInfoTopicName'] = topic_info_base + name
    plugin_dict['pointCloudTopicName'] = topic_base + name + '_points'
    plugin_dict['frame_name'] = 'ir_' + name
    plugin_dict['pointCloudCutoff'] = 0.005

    #Create camera sensor 
    create_camera_sensor(tof,camera_type='depth',plugin_type='libgazebo_ros_openni_kinect.so',
        plugin_att=plugin_dict,horizontal_fov=horizontal_fov,
        image_att=capture, clip_att=clip,attributes=dict()
        )

    joint = SubElement(el,'joint',{'name':'depth_camera_joint_'+name,'type':'fixed'})
    add_tag_dict(joint,{'child':'depth_camera_ros::link_'+name,'parent':'chassis'})

def extend_dr_drift(base, data, cam_horizontal_fov, depth_cam_horizontal_fov):
    tree = ET.parse(base)
    root = tree.getroot()

    model = root.find("model")

    # Camera mount angle
    cam_angle = data['front_camera']['angle']
    # Camera position in vehicle coordinate system
    cam_translation = np.array(data['front_camera']['translation'])

    front_camera_dict = dict()
    front_camera_dict['pose'] = np.append(np.array(data['front_camera']['translation']),
[0,data['front_camera']['angle'], 0])
    front_camera_dict['size'] = data['front_camera']['size']
    front_camera_dict['horizontal_fov'] = cam_horizontal_fov
    front_camera_dict['clip'] = data['front_camera']['clip']
    front_camera_dict['capture'] = data['front_camera']['capture']
    front_camera_dict['ros'] = data['front_camera']['ros']
    front_camera_dict['attributes'] = {'update_rate': data['front_camera']['update_rate']}

    create_front_camera(model,**front_camera_dict)

    depth_camera_dict = dict()
    depth_camera_dict['pose'] = np.append(np.array(data['depth_camera']['translation']),
[0,data['depth_camera']['angle'], 0])
    depth_camera_dict['size'] = data['depth_camera']['size']
    depth_camera_dict['horizontal_fov'] = depth_cam_horizontal_fov
    depth_camera_dict['clip'] = data['depth_camera']['clip']
    depth_camera_dict['capture'] = data['depth_camera']['capture']
    depth_camera_dict['ros'] = data['depth_camera']['ros']
    depth_camera_dict['attributes'] = {'update_rate': data['depth_camera']['update_rate']}

    create_depth_camera(model,**depth_camera_dict)

    sensors = data['time_of_flight']['sensors']

    del data['time_of_flight']['sensors']

    for name in sensors:
        pose = sensors[name]
        create_tof_camera(model,name,pose,**data['time_of_flight'])

    return prettify(root)
#%%
