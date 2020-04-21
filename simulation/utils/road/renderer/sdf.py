from road.renderer import groundplane, obstacle, traffic_sign, special_objects

# we assume that the road width config set here is the same used during the generation
from road import schema
from os import path
import os


def generate_sdf(
    xml_content,
    target_dir,
    generator_dir,
    road_name,
    add_vehicle,
    extract_tiles=False,
    tile_size=4,
    background=False,
    segmentation=False,
    fast_physics=False,
):
    """Create world.sdf file and material files for Gazebo.

    @param xml_content An XML document.  This should be data (Python 2
    str or Python 3 bytes), or a text (Python 2 unicode or Python 3
    str) in the L{pyxb._InputEncoding} encoding.

    @param target_dir Directory in which the world.sdf file should be written

    @param generator_dir Directory which contains road_generation

    @param road_name name of road to render

    @param add_vehicle Specifies if a vehicle should be added to the Gazebo world file

    @param tile_size Size of quadratic tiles measured in meters
    """

    doc = schema.CreateFromDocument(xml_content)

    #

    # Draw groundplane elements into materials directory. / Returns array of tile models
    tiles = groundplane.draw(
        doc,
        generator_dir,
        road_name=road_name,
        tile_size=tile_size,
        include_empty=not extract_tiles,
        segmentation=segmentation,
    )

    content = "".join(tiles)

    if add_vehicle:
        # Includes dr_drift into world.sdf
        content += dr_drift()
    for obst in doc.obstacle:
        if obst.type != "blockedArea":
            content += obstacle.draw(obst)
    for sign in doc.trafficSign:
        content += traffic_sign.draw(sign, target_dir)
    for ramp in doc.ramp:
        content += special_objects.draw_ramp(
            ramp.centerPoint.x, ramp.centerPoint.y, ramp.orientation, ramp.id
        )

    if background:
        content += bg_wall()

    # Write content to world.sdf in the target directory
    with open(path.join(target_dir, "model.sdf"), "w+") as file:
        file.write("<?xml version='1.0'?>\n")
        file.write("<sdf version='1.6'>\n<world name='default'>")
        file.write("")
        file.write("<wind/>")
        file.write("<atmosphere type='adiabatic'/>")
        file.write(physics(fast=fast_physics))
        file.write(scene())
        file.write(sun_light())
        file.write(content)
        file.write("</world>\n</sdf>")

    with open(path.join(target_dir, "model.config"), "w+") as file:
        file.write(config(road_name))

    with open(path.join(target_dir, ".gitignore"), "w+") as file:
        file.write(git_ignore())

    # If tiles are supposed to be extracted
    if extract_tiles:
        tiles_db = path.join(
            os.environ.get("KITCAR_REPO_PATH"),
            "kitcar-gazebo-simulation",
            "simulation",
            "models",
            "tiles_db",
        )
        idx = 0

        road_name = road_name.split("/")[-1]

        for t in tiles:
            p = path.join(tiles_db, road_name + str(idx))
            os.makedirs(p, exist_ok=True)

            with open(path.join(p, "model.sdf"), "w+") as file:

                file.write("<?xml version='1.0'?>\n")
                file.write("<sdf version='1.6'>")
                file.write(t)
                file.write("</sdf>")

            with open(path.join(p, "model.config"), "w+") as file:
                file.write(config(road_name + str(idx)))

            idx += 1


def physics(fast=False):
    gravity = "0 0 -9.81"
    contacts = 10
    step = 0.001
    iters = 50

    if fast:
        gravity = "0"
        contacts = 2
        step = 0.001
        iters = 1

    return f"""
    <gravity>{gravity}</gravity>
    <physics type="ode">
      <max_step_size>{step}</max_step_size>
      <real_time_update_rate>1000</real_time_update_rate>
      <max_contacts>{contacts}</max_contacts>
      <ode>
        <solver>
          <type>quick</type>
          <iters>{iters}</iters>
          <sor>1.4</sor>
        </solver>
        <constraints>
          <cfm>0</cfm>
          <erp>1</erp>
          <contact_max_correcting_vel>0</contact_max_correcting_vel>
          <contact_surface_layer>0</contact_surface_layer>
        </constraints>
      </ode>
    </physics>
    """


def scene():
    return """
    <scene>
      <ambient>0.7 0.7 0.7 1</ambient>
      <background>0.1 0.1 0.1 1</background>
      <shadows>0</shadows>
    </scene>
    """


def sun_light():
    return """
    <light name='sun_light' type='directional'>
      <pose frame=''>0 0 10 0 -0 0</pose>
      <diffuse>0.5 0.5 0.5 1</diffuse>
      <specular>0.1 0.1 0.1 1</specular>
      <direction>0.1 0.1 -0.9</direction>
      <attenuation>
        <range>20</range>
        <constant>0.5</constant>
        <linear>0.01</linear>
        <quadratic>0.001</quadratic>
      </attenuation>
      <cast_shadows>0</cast_shadows>
    </light>
    """


def dr_drift():
    return """
    <include>
        <uri>model://dr_drift</uri>
        <pose>0.4 -0.2 0.01 0 0 0</pose>
    </include>
    """


def bg_wall():
    return """
    <include>
        <uri>model://bg_wall</uri>
        <pose>0 0 0 0 0 0</pose>
    </include>
    """


def config(road_name):
    return """<?xml version='1.0'?>
<model>
    <name>{road_name}</name>
    <version>1.0</version>
    <sdf version='1.6'>model.sdf</sdf>
    <author>
        <name>Generated by road_generation (maintained by KITcar_simulation subteam)</name>
        <email>kditschuneit@icloud.com</email>
    </author>
    <description>Auto-generated by Commonroad road-generation</description>
</model>""".format(
        road_name=road_name
    )


def git_ignore():
    return """materials\nmodel.sdf\nmodel.config\nlines.csv\ncommonroad.xml"""
