def draw_ramp(x, y, orientation, name):
    return """
    <model name='Ramp-{name}'>
      <link name='link'>
        <visual name='visual'>
          <cast_shadows>1</cast_shadows>
          <geometry>
            <mesh>
                <uri>file://meshes/Ramp.dae</uri>
                <scale>0.001 0.001 0.001</scale>
            </mesh>
          </geometry>
        </visual>
        <collision name='collision'>
            <geometry>
                <mesh>
                    <uri>file://meshes/Ramp.dae</uri>
                    <scale>0.001 0.001 0.001</scale>
                </mesh>
            </geometry>
        </collision>
        <self_collide>0</self_collide>
      </link>
      <pose frame=''>{x} {y} {z} 0 0 {orientation}</pose>
      <static>true</static>
    </model>
    """.format(x=x, y=y, z=0.0, orientation=orientation, name=name)