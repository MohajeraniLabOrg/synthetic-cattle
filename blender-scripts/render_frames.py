import bpy, os, sys, datetime, json
C = bpy.context
D = bpy.data
O = bpy.ops

scripts_path = r'\\wsl.localhost\Ubuntu\home\galiold\projects\mmpose-synthetic-tune\blender-scripts'

if scripts_path not in sys.path:
    sys.path.append(scripts_path)

from constants import categories
from utils import hide_scene_collection, setup_scene, render_frame, change_cow_material, export_coco_annotations, change_scene

with open(f'{scripts_path}/scene_settings.json') as f:
    scene_settings = json.loads(f.read())

init_time = datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
out_path = f"{os.path.dirname(D.filepath)}/renders/{init_time}"
os.makedirs(out_path)


if __name__ == '__main__':
    images = []
    annotations = []
    ann_id = 0

    # change_scene(scene_settings, 4)

    for idx, settings in enumerate(scene_settings):
        setup_scene(settings)

        for mat in settings['materials']:
            change_cow_material(mat)
            for f in settings['frames']:
                filename = f"{idx+1}-synthetic-{settings['collection']}-{mat}-img{f:03d}.png"
                filepath = f"{out_path}/{filename}"
                render_frame(f, filepath, skip_save=True)
                img, ann = export_coco_annotations(ann_id, filename)
                images.append(img)
                annotations.append(ann)
                ann_id += 1
            
        hide_scene_collection(settings)
    
    dataset = {
        'images': images,
        'annotations': annotations,
        'categories': categories
    }

    with open(f'{out_path}/synthetic-{init_time}.json', 'w') as f:
        f.write(json.dumps(dataset))