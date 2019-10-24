import shutil, pkg_resources, os, math
import numpy as np

def draw(target_dir, lanelets):
    model_file = "car-cc2017.dae"
    model_stream = pkg_resources.resource_stream("commonroad.renderer.models",
        model_file)
    with open(os.path.join(target_dir, 'meshes', model_file), "wb") as model_target:
        shutil.copyfileobj(model_stream, model_target)

    return """
    <model name="EgoVehicle">
      <pose>0 0 0  0 0 0</pose>
      <link name="body">
        <visual name="visual">
          <geometry>
            <mesh><uri>file://{0}</uri></mesh>
          </geometry>
        </visual>
        <collision name="collision">
          <pose>0.11 0 0.04  0 0 0</pose>
          <geometry>
            <box>
              <size>0.3 0.2 0.08</size>
            </box>
          </geometry>
        </collision>
        <sensor type="camera" name="my_sensor">
          <pose>0 0 0.32 0 0.7 0</pose>
          <camera>
            <horizontal_fov>2.0</horizontal_fov>
            <image>
              <width>1280</width>
              <height>920</height>
            </image>
            <clip>
              <near>0.1</near>
              <far>100</far>
            </clip>
            <save enabled="true"><path>/tmp/driving</path></save>
            <lens>
              <type>gnomonical</type>
              <scale_to_hfov>true</scale_to_hfov>
              <cutoff_angle>2.4</cutoff_angle>
              <env_texture_size>512</env_texture_size>
            </lens>
          </camera>
          <always_on>1</always_on>
          <update_rate>100</update_rate>
          <visualize>true</visualize>
        </sensor>
      </link>
      <plugin name="keyframes" filename="libkeyframes.so">
        {1}
      </plugin>
    </model>
    """.format(model_file, compute_keyframes(lanelets))

def compute_keyframes(lanelets):
    current_lanelet = get_start_lanelet(lanelets)
    keyframes = ""
    t = 0
    last_point = None
    while current_lanelet is not None:
        middle = middle_of_lanelet(current_lanelet)
        for p in middle:
            if last_point is not None:
                orientation = math.atan2(p[1] - last_point[1], p[0] - last_point[0])
            else:
                orientation = 0
            keyframes += '<keyframe t="{t}" x="{x}" y="{y}" z="{z}" o="{o}" />\n'.format(
                t=t, x=p[0], y=p[1], z=0, o=orientation)
            if last_point is not None:
                dx = last_point[0] - p[0]
                dy = last_point[1] - p[1]
                t += math.sqrt(dx*dx + dy*dy)
            last_point = p
        current_lanelet = get_next_lanelet(lanelets, current_lanelet)
    return keyframes

def boundary_point_lengths(boundary):
    result = [0]
    len = 0
    for (p1, p2) in zip(boundary.point, boundary.point[1:]):
        len += distance_points(p1, p2)
        result.append(len)
    return result

def boundary_to_equi_distant(boundary):
    lengths = boundary_point_lengths(boundary)
    x = list(map(lambda p: p.x, boundary.point))
    y = list(map(lambda p: p.y, boundary.point))
    STEPS = 20
    eval_marks = np.arange(0, lengths[-1], lengths[-1]/STEPS)
    xinterp = np.interp(eval_marks, lengths, x)
    yinterp = np.interp(eval_marks, lengths, y)
    return map(lambda i: (i[0],i[1]), zip(xinterp, yinterp))

def middle_of_lanelet(lanelet):
    left = boundary_to_equi_distant(lanelet.leftBoundary)
    right = boundary_to_equi_distant(lanelet.rightBoundary)
    return list(map(lambda p: ((p[0][0] + p[1][0])/2, (p[0][1] + p[1][1])/2),
        zip(left, right)))

def get_lanelet_by_id(lanelet_list, id):
    for lanelet in lanelet_list:
        if lanelet.id == id:
            return lanelet
    return None

def get_start_lanelet(lanelets):
    for lanelet in lanelets:
        if lanelet.isStart:
            return lanelet
    return None

def get_next_lanelet(lanelets, ll):
    for x in ll.successor.lanelet:
        return get_lanelet_by_id(lanelets, x.ref)
    return None

def distance_points(p1, p2):
    dx = p2.x - p1.x
    dy = p2.y - p1.y
    return math.sqrt(dx*dx + dy*dy)
