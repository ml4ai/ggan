"""
Renders the shapenet obj files from num_views, can modify the camera location, light position
Currently supports: euler angles for rotation: later add support for quaternions
"""

# blender --background --python mytest.py -- --views 10 /path/to/my.obj
# Ref: https://github.com/panmari/stanford-shapenet-renderer

import os, math
import bpy

num_views = 10
obj_path = "/home/user/data/car_full/02958343/86c8a3137a716d70e742b0b5e87bec54/models/model_normalized.obj"
scale = 1.0
remove_doubles = True
edge_split = True
color_depth = "16"
resolution = 1024
engine = "BLENDER_EEVEE"

# Set up rendering
context = bpy.context
scene = bpy.context.scene
render = bpy.context.scene.render

render.engine = engine
render.image_settings.color_mode = 'RGBA'  # ('RGB', 'RGBA', ...)
render.image_settings.color_depth = color_depth  # ('8', '16')
render.resolution_x = resolution
render.resolution_y = resolution
render.resolution_percentage = 100
render.film_transparent = True

scene.use_nodes = True
scene.view_layers["View Layer"].use_pass_normal = True
scene.view_layers["View Layer"].use_pass_diffuse_color = True
scene.view_layers["View Layer"].use_pass_object_index = True

nodes = bpy.context.scene.node_tree.nodes
links = bpy.context.scene.node_tree.links

# Clear default nodes
for n in nodes:
    nodes.remove(n)

# Create input render layer node
render_layers = nodes.new('CompositorNodeRLayers')

# Create id map output nodes
id_file_output = nodes.new(type="CompositorNodeOutputFile")
id_file_output.label = 'ID Output'
id_file_output.base_path = ''
id_file_output.file_slots[0].use_node_format = True
id_file_output.format.file_format = "PNG"
id_file_output.format.color_depth = color_depth

# Delete default cube
context.active_object.select_set(True)
bpy.ops.object.delete()

# Import textured mesh
bpy.ops.object.select_all(action='DESELECT')

bpy.ops.import_scene.obj(filepath=obj_path)

obj = bpy.context.selected_objects[0]
context.view_layer.objects.active = obj

# Set objekt IDs
obj.pass_index = 1

# Make light just directional, disable shadows.
light = bpy.data.lights['Light']
light.type = 'SUN'
light.use_shadow = False
# Possibly disable specular shading:
light.specular_factor = 1.0
light.energy = 10.0

# Add another light source so stuff facing away from light is not completely dark
bpy.ops.object.light_add(type='SUN')
light2 = bpy.data.lights['Sun']
light2.use_shadow = False
light2.specular_factor = 1.0
light2.energy = 0.015
bpy.data.objects['Sun'].rotation_euler = bpy.data.objects['Light'].rotation_euler
bpy.data.objects['Sun'].rotation_euler[0] += 180

# Place camera
cam = scene.objects['Camera']
cam.location = (1.7321, 0.0000, 1.0000)
cam.data.lens = 35
cam.data.sensor_width = 32

cam_constraint = cam.constraints.new(type='TRACK_TO')
cam_constraint.track_axis = 'TRACK_NEGATIVE_Z'
cam_constraint.up_axis = 'UP_Y'

cam_empty = bpy.data.objects.new("Empty", None)
cam_empty.location = (0, 0, 0)
cam.parent = cam_empty

scene.collection.objects.link(cam_empty)
context.view_layer.objects.active = cam_empty
cam_constraint.target = cam_empty

stepsize = 360.0 / num_views
rotation_mode = 'XYZ'

model_identifier = os.path.split(os.path.split(obj_path)[0])[1]
fp = os.path.join(os.path.abspath("./output"), model_identifier, model_identifier)

for i in range(0, num_views):
    print("Rotation {}, {}".format((stepsize * i), math.radians(stepsize * i)))

    render_file_path = fp + '_r_{0:03d}'.format(int(i * stepsize))

    scene.render.filepath = render_file_path
    id_file_output.file_slots[0].path = render_file_path + "_id"

    bpy.ops.render.render(write_still=True)  # render still

    cam_empty.rotation_euler[2] += math.radians(stepsize)
