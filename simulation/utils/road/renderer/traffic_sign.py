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
                <scale>1 1 1</scale>
            </mesh>
        </geometry>
        </visual>
        <collision name='collision'>
            <pose>{box_position_x} {box_position_y} {box_position_z} 0 0 0</pose>
            <geometry>
                <box>
                <size>{box_size_x} {box_size_y} {box_size_z}</size>
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
        box_position_x=collision_box_position[0],
        box_position_y=collision_box_position[1],
        box_position_z=collision_box_position[2],
        box_size_x=collision_box_size[0],
        box_size_y=collision_box_size[1],
        box_size_z=collision_box_size[2],
    )
