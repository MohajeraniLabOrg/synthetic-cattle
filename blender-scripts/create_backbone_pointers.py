import bpy

C = bpy.context
O = bpy.data.objects
D = bpy.data
vert = bpy.context.object.data.vertices
face = bpy.context.object.data.polygons #not faces!
edge = bpy.context.object.data.edges
obj = C.active_object
mesh = obj.to_mesh()

# Function to clear selected vertices
def clear_vertex_select():
    for i in face:                   
        i.select=False               
    for i in edge:
        i.select=False
    for i in vert:
        i.select=False

marker_pts = list(map(lambda v: v.index, filter(lambda v: v.select, obj.data.vertices)))

clear_vertex_select()


try:
    col = D.collections['markers']
except:
    col = D.collections.new("markers")
    C.scene.collection.children.link(col)

for idx, v in enumerate(marker_pts):
    e = O.new(f'bptr-{idx}', None)
    e.location = mesh.vertices[v].co
    col.objects.link(e)
    C.active_object.data.vertices[v].select = True
    e.select_set(True)
    bpy.ops.object.mode_set(mode='EDIT')
    bpy.ops.object.vertex_parent_set()
    bpy.ops.object.mode_set(mode='OBJECT')
    C.active_object.data.vertices[v].select = False
    e.select_set(False)
#    e.hide_set(True)
    
