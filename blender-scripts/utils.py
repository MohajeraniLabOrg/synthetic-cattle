# Utility functions to be used in blender Python Console
import bpy, os, sys, datetime, json, bpy_extras, copy
from constants import ann_template, bones, img_template
C = bpy.context
D = bpy.data
O = bpy.ops

render_size_x = C.scene.render.resolution_x
render_size_y = C.scene.render.resolution_y


def rebuild_softbody_cache():
    with bpy.context.temp_override( point_cache=D.objects['BlackCattle_Body'].modifiers['Softbody'].point_cache ):
        bpy.ops.ptcache.free_bake()
        bpy.ops.ptcache.bake_all(bake=True)


def rebuild_hair_cache():
    with bpy.context.temp_override(
        point_cache=D.objects['BlackCattle_Body'].particle_systems['BlackCattle_Hair_Tail_Long'].point_cache):
        bpy.ops.ptcache.free_bake()
        bpy.ops.ptcache.bake(bake=True)


def show_scene_collection(settings):
    D.collections[settings['collection']].hide_viewport = False
    D.collections[settings['collection']].hide_render = False


def hide_scene_collection(settings):
    D.collections[settings['collection']].hide_viewport = True
    D.collections[settings['collection']].hide_render = True


def adjust_cow_root(settings, bake=True):
    D.objects['BlackCattle_Rig_grp'].pose.bones['Root'].location = settings['cow_root']['location']
    D.objects['BlackCattle_Rig_grp'].pose.bones['Root'].rotation_quaternion = settings['cow_root']['rotation_quaternion']
    if bake:
        rebuild_softbody_cache()
        rebuild_hair_cache()


def adjust_camera(settings):
    D.objects['Camera'].rotation_euler = settings['camera']['rotation_euler']
    D.objects['Camera'].location = settings['camera']['location']
    D.cameras['Camera'].background_images[0].image = D.images.load(settings['background']['file'])


def adjust_background(settings):
    C.scene.node_tree.nodes['Background'].image = D.images.load(settings['background']['file'])


def adjust_world(settings):
    D.worlds['World'].node_tree.nodes['Environment Texture'].image = D.images.load(settings['env']['file'])
    D.worlds['World'].node_tree.nodes['Mapping'].inputs['Rotation'].default_value = settings['env']['rotation_euler']
    D.worlds['World'].node_tree.nodes['Background'].inputs['Strength'].default_value = settings['env']['strength']
    

def setup_scene(settings, bake=True):
    show_scene_collection(settings)
    adjust_cow_root(settings, bake)
    adjust_background(settings)
    adjust_camera(settings)
    adjust_world(settings)


def change_scene(scene_settings, scene_num):
    try:
        hide_scene_collection(scene_settings[scene_num - 1])
        setup_scene(scene_settings[scene_num], bake=False)
    except:
        pass


def change_cow_material(material):
    D.objects['BlackCattle_Body'].material_slots[0].material = D.materials[f'BlackCattle_Body_{material}']
    D.objects['BlackCattle_Body'].material_slots[1].material = D.materials[f'BlackCattle_Hair_Body_{material}']
    D.objects['BlackCattle_Body'].material_slots[2].material = D.materials[f'BlackCattle_Hair_Head_{material}']


def render_frame(frame_number, filepath, skip_save=False):
    C.scene.frame_set(frame_number)
    if skip_save:
        return
    C.scene.render.filepath = filepath
    bpy.ops.render.render(write_still=True)


def get_scene_properties():
    background = D.cameras['Camera'].background_images[0].image.filepath_from_user()
    base_name: str = os.path.basename(background)
    file_name = base_name.split('.')[0]
    scene_settings = {
        'collection': f'scene{file_name}',
        'cow_root': {
            'location': list(D.objects['BlackCattle_Rig_grp'].pose.bones['Root'].location),
            'rotation_quaternion': list(D.objects['BlackCattle_Rig_grp'].pose.bones['Root'].rotation_quaternion)
        },
        "cow_parameters": {
            "materials": [
                "Original",
                "Brown",
                "White",
                "Original_White",
                "Brown_White"
            ]
        },
        'background': {
            'file': D.cameras['Camera'].background_images[0].image.filepath_from_user()
        },
        'env': {
            'file': D.worlds['World'].node_tree.nodes['Environment Texture'].image.filepath_from_user(),
            'rotation_euler': list(D.worlds['World'].node_tree.nodes['Mapping'].inputs['Rotation'].default_value),
            'strength': D.worlds['World'].node_tree.nodes['Background'].inputs['Strength'].default_value
        },
        'camera': {
            'location': list(D.objects['Camera'].location),
            'rotation_euler': list(D.objects['Camera'].rotation_euler)
        }
    }
    print(json.dumps(scene_settings))


def calculate_bbox(Xs, Ys):
    x_min = min(Xs)
    x_max = max(Xs)
    y_min = min(Ys)
    y_max = max(Ys)
    w = x_max - x_min
    h = y_max - y_min
    
    return (x_min, y_min, w, h)


def get_world_coords(object_name):


    object_data = D.objects[object_name]
    pos = object_data.matrix_world.to_translation()
    coords2d = bpy_extras.object_utils.world_to_camera_view(C.scene, D.objects['Camera'], pos)
    
    x = round(render_size_x * coords2d.x)
    y = round(render_size_y * coords2d.y)
    vis = 2
    
    if (x <= 0 or y <= 0) or (x > render_size_x or y > render_size_y):
        x = vis = 0
        y = render_size_y
        
    return x, y, vis


def export_coco_annotations(idx, filename):
    C.view_layer.update()

    img = copy.deepcopy(img_template)
    img['id'] = idx
    img['width'] = render_size_x
    img['height'] = render_size_y
    img['file_name'] = filename

    ann = copy.deepcopy(ann_template)
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

    return img, ann
