import bpy_extras
import bpy
import copy as cp
import json
from os import path
from datetime import datetime as dt
import sys

scripts_path = r'C:\Users\Ali Goldani\projects-win\blender\Blender_Reqs\blender-scripts'

if scripts_path not in sys.path:
    sys.path.append(scripts_path)

from constants import ann_template, bones, categories, img_template, frames_to_render

C = bpy.context
O = bpy.data.objects
obj =  C.active_object

scene = bpy.context.scene
cam = bpy.data.objects['Camera']

render_size_x = scene.render.resolution_x
render_size_y = scene.render.resolution_y


time = dt.now().strftime('%d-%m-%Y-%H-%M-%S')

file_path = bpy.data.filepath
export_notes = f"{time}-render_test"
annotation_path = f'{path.dirname(file_path)}/annotations/render-{export_notes}.json'


####### Functions

def calculate_bbox(Xs, Ys):
    x_min = min(Xs)
    x_max = max(Xs)
    y_min = min(Ys)
    y_max = max(Ys)
    w = x_max - x_min
    h = y_max - y_min
    
    return (x_min, y_min, w, h)


def get_world_coords(object_name):
    object_data = O[object_name]
    pos = object_data.matrix_world.to_translation()
    coords2d = bpy_extras.object_utils.world_to_camera_view(scene, cam, pos)
    
    x = round(render_size_x * coords2d.x)
    y = round(render_size_y * coords2d.y)
    vis = 2
    
    if (x <= 0 or y <= 0) or (x > render_size_x or y > render_size_y):
        x = vis = 0
        y = render_size_y
        
    return x, y, vis


####### Exporting in COCO format


images = []
annotations = []

for idx in frames_to_render:
    scene.frame_current = idx
    C.view_layer.update()

    img = cp.deepcopy(img_template)
    img['id'] = idx
    img['width'] = render_size_x
    img['height'] = render_size_y
    img['file_name'] = f'render-{export_notes}-img{idx:03d}.png'
    
    ann = cp.deepcopy(ann_template)
    ann['id'] = idx
    ann['image_id'] = idx
    
    keypoints = []
    Xs = []
    Ys = []
    
    for bone in bones:
        x, y, vis = get_world_coords(bone)

        keypoints.extend([ x, render_size_y - y, vis ])
        Xs.append(x)
        Ys.append(y)
    
    ann['bbox'] = calculate_bbox(Xs, Ys)
    ann['keypoints'] = keypoints
    
    images.append(img)
    annotations.append(ann)
    
    
dataset = {
    'images': images,
    'annotations': annotations,
    'categories': categories
}

with open(annotation_path, 'w') as f:
    f.write(json.dumps(dataset))