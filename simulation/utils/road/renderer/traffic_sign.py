from simulation.utils.road.sections import TrafficSign


def draw(name: str, sign: TrafficSign):
    return model(
        sign.center.x,
        sign.center.y,
        0.0,
        sign.orientation,
        name,
        sign.kind.mesh,
        sign.kind.collision_box_position,
        sign.kind.collision_box_size,
    )


def model(x, y, z, orientation, name, mesh, collision_box_position, collision_box_size):
    return """
    <model name='{name}'>
      <static>1</static>
      <link name='link'>
        <visual name='visual'>
          <cast_shadows>1</cast_shadows>
          <geometry>
            <mesh>
                <uri>file://meshes/{mesh}.dae</uri>
                <scale>0.001 0.001 0.001</scale>
            </mesh>
          </geometry>
        </visual>
        <collision name='collision'>
            <pose>{collision_box_position} 0 0 0</pose>
            <geometry>
                <box>
                  <size>{collision_box_size}</size>
                </box>
            </geometry>
        </collision>
        <self_collide>0</self_collide>
      </link>
      <pose frame=''>{x} {y} {z} {angle} 0 {orientation}</pose>
    </model>
    """.format(
        x=x,
        y=y,
        z=z,
        orientation=orientation,
        name=name,
        mesh=mesh,
        angle=0,
        collision_box_position=collision_box_position,
        collision_box_size=collision_box_size,
    )
