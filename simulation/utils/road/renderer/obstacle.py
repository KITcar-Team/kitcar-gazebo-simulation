def draw(name, obst):
    return obstacle_model(
        name,
        obst.center.x,
        obst.center.y,
        obst.depth,
        obst.width,
        obst.height,
        obst.orientation,
    )


# large masses make model instable
def obstacle_model(name, x, y, length, width, height, orientation):
    return """
    <model name='{name}'>
      <pose frame=''>{x} {y} {z} 0 0 {orientation}</pose>
      <link name='link'>
        <inertial>
          <mass>1</mass>
          <inertia>
            <ixx>0.166667</ixx>
            <ixy>0</ixy>
            <ixz>0</ixz>
            <iyy>0.166667</iyy>
            <iyz>0</iyz>
            <izz>0.166667</izz>
          </inertia>
        </inertial>
        <collision name='collision'>
          <geometry>
            <box>
              <size>{length} {width} {height}</size>
            </box>
          </geometry>
          <max_contacts>10</max_contacts>
          <surface>
            <contact>
              <ode/>
            </contact>
            <bounce/>
            <friction>
              <torsional>
                <ode/>
              </torsional>
              <ode/>
            </friction>
          </surface>
        </collision>
        <visual name='visual'>
          <geometry>
            <box>
              <size>{length} {width} {height}</size>
            </box>
          </geometry>
          <material>
            <script>
              <name>Gazebo/White</name>
              <uri>file://media/materials/scripts/gazebo.material</uri>
            </script>
          </material>
        </visual>
        <self_collide>0</self_collide>
        <enable_wind>0</enable_wind>
        <kinematic>0</kinematic>
      </link>
    </model>
    """.format(
        name=name,
        x=x,
        y=y,
        z=height / 2 + 0.1,
        length=length,
        width=width,
        height=height,
        orientation=orientation,
    )
