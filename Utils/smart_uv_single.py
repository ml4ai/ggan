"""
blender python code: that loads given obj model, does smart uv projection and saves the new obj file
that has vertex to uv mapping relation
run from terminal: blender --background --python smart_uv_single.py
"""

import bpy

# obj load path
file_path = "/home/user/data/shapenet_car/02958343/1a0bc9ab92c915167ae33d942430658c/models/model_normalized.obj"
# obj save path
save_path = "/home/user/data/shapenet_car/02958343/1a0bc9ab92c915167ae33d942430658c/models/bpy_model.obj"

print("\nRemoving Cube/Camera/Light from scene")
# Remove the existing Cube/Camera/Light
for temp_obj in bpy.context.scene.objects:
    temp_obj.select_set(True)
# delete unselected objects
bpy.ops.object.delete()

print("\nLoading obj file")
# load obj file
imported_object = bpy.ops.import_scene.obj(filepath=file_path)
obj = bpy.context.selected_objects[0]
print(f"Imported name: {obj.name}")

print("\nPerforming smart uv projection...")
# perform smart uv projection
bpy.context.view_layer.objects.active = obj
# enter into edit mode from object mode
bpy.ops.object.editmode_toggle()
# select all vertices of the object
bpy.ops.mesh.select_all(action="SELECT")
# perform smart uv projection and unwrapping
bpy.ops.uv.smart_project(angle_limit=66.0, island_margin=0.3, user_area_weight=0.0, use_aspect=True, stretch_to_bounds=False)
# exit from edit mode to object mode
bpy.ops.object.editmode_toggle()

print("\nsaving new obj file")
# save the new obj file that has uv mapping information
bpy.ops.export_scene.obj(filepath=save_path)
