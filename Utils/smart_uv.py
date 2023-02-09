"""
blender python code: that loads given obj model, does smart uv projection and saves the new obj file
that has vertex to uv mapping relation
run from terminal: blender --background --python smart_uv.py
it runs smart uv projection for all obj files inside a given folder and
saves bpy_model.obj for at the same place as original obj file
Uses multiprocessing to run multiple parallel processes and speed up things
"""

import os
import multiprocessing

source_folder = "/home/user/data/car_full/02958343"
folders = os.listdir(source_folder)

# use multiprocessing to speed up the smart uv projection
# get list of obj files without corresponding bpy_model.obj file
# chunks to process within a single process
n = 10

unprocessed = []
for folder in folders:
    obj_path = source_folder + "/" + folder + "/models/model_normalized.obj"
    # some of them do not have obj file [don't know the reason]
    if os.path.exists(obj_path):
        # check if it's already processed and has corresponding bpy_model.obj file
        bpy_model = source_folder + "/" + folder + "/models/bpy_model.obj"
        # if it exists, we don't need to process it, else add the obj path to unprocessed list
        if not os.path.exists(bpy_model):
            unprocessed.append(obj_path)

if len(unprocessed) == 0:
    print("every obj files processed.")
    exit()


def generator_from_list(lst, n_size):
    # generate n sized chunk from a given list
    for i in range(0, len(lst), n_size):
        yield lst[i: i + n_size]


def smart_uv_projection(path_obj):
    import bpy

    print("\nRemoving Cube/Camera/Light from scene")
    # Remove the existing Cube/Camera/Light
    for temp_obj in bpy.context.scene.objects:
        temp_obj.select_set(True)
    # delete selected objects
    bpy.ops.object.delete()

    print(f"\nProcessing: {path_obj}")
    # obj save path
    save_path = path_obj.rsplit("/", 1)[0] + "/bpy_model.obj"

    print("\nLoading obj file")
    bpy.ops.import_scene.obj(filepath=path_obj)
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

    # remove the object
    print("Deleting old object")
    for t_obj in bpy.context.scene.objects:
        t_obj.select_set(True)
    # delete selected objects
    bpy.ops.object.delete()


manager = multiprocessing.Manager()
jobs = []
for idx, obj_path in enumerate(unprocessed):
    p = multiprocessing.Process(target=smart_uv_projection, args=(obj_path,))
    jobs.append(p)

counter = 0
for chunks in generator_from_list(jobs, n):
    for proc in chunks:
        proc.start()

    for p in chunks:
        p.join()

    counter += n
    print(f"models processed: {counter}")

    # close the open processes
    for p in chunks:
        p.close()
