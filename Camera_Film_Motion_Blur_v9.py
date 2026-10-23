# SPDX-License-Identifier: GPL-3.0-or-later
bl_info = {
    "name": "Camera Film Motion Blur",
    "author": "Claude",
    "version": (1, 6, 0),
    "blender": (4, 2, 0),
    "location": "Properties > Camera Data > Film Motion Blur",
    "description": "Camera attribute that adds physically based motion blur for film "
                   "running through a projector (no shutter; pure eye/exposure smear)",
    "category": "Render",
}

import math

import numpy as np
import bpy
from bpy.app.handlers import persistent
from bpy.props import BoolProperty, EnumProperty, FloatProperty, PointerProperty
from bpy.types import Operator, Panel, PropertyGroup

NODE_NAME = "CameraFilmMotionBlur"


# ----------------------------------------------------------------------------
# Physics
#
# Film running at constant speed has no shutter to freeze it, so every point of
# the picture is smeared along the direction of travel by
#
#     blur length = image speed x integration time
#
# image speed      = film_speed (frames/s) x frame height on screen
# integration time = exposure time of the "sensor" (the eye: ~40 ms, about one
#                    frame at 24 fps, is a good perceptual figure)
#
# Light collected over that time is a uniform average along the travel path,
# i.e. a box (not Gaussian) blur. Length is expressed as a fraction of the
# camera image height (vertical travel) or width (horizontal travel).
# ----------------------------------------------------------------------------
def blur_fraction(s):
    """Blur length as a fraction of the image dimension along the film's travel."""
    return max(0.0, s.film_speed * (s.integration_ms / 1000.0) * (s.frame_size / 100.0))


# ----------------------------------------------------------------------------
# Compositor helpers (4.x scene.node_tree and 5.x scene.compositing_node_group)
# ----------------------------------------------------------------------------
def _get_tree(scene, create=False):
    if hasattr(scene, "compositing_node_group"):          # Blender 5.x
        tree = scene.compositing_node_group
        if tree is None and create:
            tree = bpy.data.node_groups.new("Film Motion Blur Compositing",
                                            "CompositorNodeTree")
            tree.interface.new_socket(name="Image", in_out='OUTPUT',
                                      socket_type='NodeSocketColor')
            rl = tree.nodes.new("CompositorNodeRLayers")
            out = tree.nodes.new("NodeGroupOutput")
            out.location = (300, 0)
            tree.links.new(rl.outputs["Image"], out.inputs[0])
            scene.compositing_node_group = tree
        return tree
    if create and not scene.use_nodes:                    # Blender 4.x
        scene.use_nodes = True
    tree = scene.node_tree
    if tree is not None and create and not any(
            n.bl_idname == "CompositorNodeComposite" for n in tree.nodes):
        rl = tree.nodes.new("CompositorNodeRLayers")
        out = tree.nodes.new("CompositorNodeComposite")
        out.location = (300, 0)
        tree.links.new(rl.outputs["Image"], out.inputs["Image"])
    return tree


OUTPUT_IDS = ("CompositorNodeComposite", "NodeGroupOutput",
              "CompositorNodeOutputFile", "CompositorNodeViewer")


def _find_output(tree):
    for idname in OUTPUT_IDS:
        for n in tree.nodes:
            if n.bl_idname == idname and n.inputs and n.inputs[0].links:
                return n
    for idname in ("CompositorNodeComposite", "NodeGroupOutput"):
        for n in tree.nodes:
            if n.bl_idname == idname:
                return n
    return None


def _set_box_filter(node):
    """Select the flat (box) kernel; its identifier differs between versions."""
    for ident in ("FLAT", "BOX"):
        try:
            if node.filter_type != ident:
                node.filter_type = ident
            return True
        except Exception:
            continue
    for sock in node.inputs:                              # in case it became a socket
        if sock.name.lower() in ("type", "filter type"):
            try:
                sock.default_value = "FLAT"
                return True
            except Exception:
                pass
    return False


def _ensure_node(scene):
    """Create (or repair) the blur node, spliced in just before the final outputs
    (after any Denoise node, so the denoiser never sees blurred pixels)."""
    tree = _get_tree(scene, create=True)
    if tree is None:
        return None
    node = tree.nodes.get(NODE_NAME)
    if node is None:
        node = tree.nodes.new("CompositorNodeBlur")
        node.name = NODE_NAME
        node.label = "Film Motion Blur (camera)"
        node.use_custom_color = True
        node.color = (0.25, 0.3, 0.45)
        for attr, val in (("use_relative", False), ("use_extended_bounds", False),
                          ("use_variable_size", False), ("use_bokeh", False),
                          ("use_gamma_correction", False)):
            try:
                setattr(node, attr, val)
            except Exception:
                pass
    _set_box_filter(node)

    out = _find_output(tree)
    if out is not None and not node.outputs[0].links:
        src = None
        if out.inputs[0].links:
            src = out.inputs[0].links[0].from_socket
        else:
            for n in tree.nodes:
                if n.bl_idname == "CompositorNodeRLayers":
                    src = n.outputs["Image"]
                    break
        if src is not None and src.node is not node:
            # Everything that currently shows/saves the final image gets the blurred one.
            targets = [l.to_socket for l in src.links
                       if l.to_node.bl_idname in OUTPUT_IDS]
            if not targets:
                targets = [out.inputs[0]]
            tree.links.new(src, node.inputs[0])
            for sock in targets:
                tree.links.new(node.outputs[0], sock)
            node.location = (src.node.location.x + 260, src.node.location.y - 120)
    scene.render.use_compositing = True
    return node


def _set_blur_size(node, sx, sy):
    """Set the box size in pixels along X / Y across Blender versions."""
    if hasattr(node, "size_x") and hasattr(node, "size_y"):
        node.size_x = int(round(sx))
        node.size_y = int(round(sy))
        return True
    for sock in node.inputs:                               # 5.x socket-based size
        if sock.name.lower() == "size" and hasattr(sock, "default_value"):
            dv = sock.default_value
            try:
                if len(dv) >= 2:
                    new = list(dv)
                    new[0], new[1] = sx, sy
                    sock.default_value = new
                    return True
            except TypeError:
                pass
    return False


# ----------------------------------------------------------------------------
# Automatic direction: follow the tracked object(s) and work out where their
# motion ends up in the camera image
#
# The thing being blurred is often NOT what the camera looks at. In a projector
# rig the film sits behind the camera and only its picture lands on a screen in
# front of it. So, when a Lens and a Screen object are given, every point of the
# tracked object is sent along the ray  film point -> lens centre -> screen plane
# (the way light really travels through a projector, which also flips the image),
# and the spot where that ray lands on the screen is projected into the camera.
# Without a Lens/Screen the points are projected straight into the camera.
#
# Each frame, the shift of every point since the previous frame is a motion
# vector. The dominant axis of those vectors (principal component, so up/down and
# left/right count the same) is the travel direction. It is smoothed over a few
# frames and held while nothing moves.
# ----------------------------------------------------------------------------
_AUTO = {"frame": None, "key": None, "prev": None, "M": None, "vec": None,
         "text": "No motion seen yet: change frame (using vertical)"}
MAX_VERTS = 3000000            # only thin out meshes bigger than this
MIN_MOTION_PX = 0.2
SNAP = 0.17                    # |component| below this is treated as exactly 0 (~10 deg)


def reset_auto():
    _AUTO.update(frame=None, key=None, prev=None, M=None, vec=None,
                 text="No motion seen yet: change frame (using vertical)")


def _mesh_world_points(ob):
    """World-space vertices of an object's own (unevaluated) mesh; origin if none."""
    mw = np.array(ob.matrix_world, dtype=np.float64)
    pts = None
    if ob.type == 'MESH' and len(ob.data.vertices):
        n = len(ob.data.vertices)
        co = np.empty(n * 3, dtype=np.float32)
        ob.data.vertices.foreach_get("co", co)
        pts = co.reshape(-1, 3).astype(np.float64)
    if pts is None:
        return mw[:3, 3][None, :]
    return pts @ mw[:3, :3].T + mw[:3, 3]


def _evaluated_world_points(ob, dg):
    ev = ob.evaluated_get(dg)
    pts = None
    try:
        me = ev.to_mesh()
        try:
            n = len(me.vertices)
            if n:
                co = np.empty(n * 3, dtype=np.float32)
                me.vertices.foreach_get("co", co)
                pts = co.reshape(-1, 3)
        finally:
            ev.to_mesh_clear()
    except Exception:
        pts = None
    mw = np.array(ev.matrix_world, dtype=np.float64)
    if pts is None or len(pts) == 0:                       # empties etc.: use the origin
        return mw[:3, 3][None, :]
    return pts.astype(np.float64) @ mw[:3, :3].T + mw[:3, 3]


def _tracked_objects(s):
    objs = []
    if s.auto_target is not None:
        objs.append(s.auto_target)
    if s.auto_collection is not None:
        for o in s.auto_collection.all_objects:
            if o not in objs:
                objs.append(o)
    return objs


def _screen_plane(screen):
    """(point on plane, unit normal) of the screen object, in world space."""
    pts = _mesh_world_points(screen)
    centre = pts.mean(axis=0)
    mw3 = np.array(screen.matrix_world, dtype=np.float64)[:3, :3]
    normal = mw3 @ np.array([0.0, 0.0, 1.0])
    try:
        if screen.type == 'MESH' and len(screen.data.polygons):
            normal = mw3 @ np.array(screen.data.polygons[0].normal, dtype=np.float64)
    except Exception:
        pass
    ln = np.linalg.norm(normal)
    return centre, (normal / ln if ln > 0 else np.array([0.0, 0.0, 1.0]))


def _project_points(scene, s, objs):
    """Tracked points in camera pixels (x right, y up): positions, in-view mask, valid mask."""
    dg = bpy.context.evaluated_depsgraph_get()
    cam = scene.camera
    scale = scene.render.resolution_percentage / 100.0
    res_x = scene.render.resolution_x * scale
    res_y = scene.render.resolution_y * scale
    world = np.concatenate([_evaluated_world_points(o, dg) for o in objs])
    world = world[::max(1, len(world) // MAX_VERTS)]
    ok = np.ones(len(world), dtype=bool)
    if s.auto_screen is not None:
        centre, normal = _screen_plane(s.auto_screen)
        if s.auto_lens is not None:
            # Line through the projection point (lens or light) and the film point,
            # extended to the screen plane. Works whether the point is a lens between
            # film and screen (image flips) or a light behind the film (no flip).
            origin = _mesh_world_points(s.auto_lens).mean(axis=0)
            d = world - origin
            denom = d @ normal
            good = np.abs(denom) > 1e-12
            t_hit = np.zeros(len(world))
            t_hit[good] = ((centre - origin) @ normal) / denom[good]
            ok = good
            world = origin + t_hit[:, None] * d
        else:
            # No lens or light: drop every film point straight onto the screen plane.
            dist = (world - centre) @ normal
            world = world - dist[:, None] * normal
    view = np.array(cam.evaluated_get(dg).matrix_world.inverted(), dtype=np.float64)
    proj = np.array(cam.calc_matrix_camera(dg, x=int(res_x), y=int(res_y)),
                    dtype=np.float64)
    h = np.concatenate([world, np.ones((len(world), 1))], axis=1)
    clip = h @ (proj @ view).T
    w = clip[:, 3]
    ok &= w > 1e-4
    ndc = np.zeros((len(world), 2))
    ndc[ok] = clip[ok, :2] / w[ok, None]
    vis = ok & (np.abs(ndc[:, 0]) <= 1.0) & (np.abs(ndc[:, 1]) <= 1.0)
    return ndc * np.array([res_x / 2.0, res_y / 2.0]), vis, ok


def _describe(ux, uy):
    ang = math.degrees(math.atan2(abs(uy), abs(ux)))      # 0 = horizontal, 90 = vertical
    if ang >= 80:
        return "Vertical"
    if ang <= 10:
        return "Horizontal"
    return "Diagonal (%d deg)" % round(ang)


def _auto_vector(scene, s):
    """Unit travel direction (ux, uy) in image pixels, or None if not known yet."""
    st = _AUTO
    objs = _tracked_objects(s)
    if not objs:
        st["text"] = "Pick an object or collection to track (using vertical)"
        return None
    key = (tuple(o.name for o in objs), s.auto_lens, s.auto_screen)
    key = (key[0], getattr(key[1], "name", None), getattr(key[2], "name", None))
    if st["key"] != key:
        reset_auto()
        st["key"] = key
    frame = scene.frame_current + scene.frame_subframe
    if st["frame"] == frame:                               # same frame again: reuse
        return st["vec"]
    try:
        pix, vis, ok = _project_points(scene, s, objs)
    except Exception as e:
        st["text"] = "Tracking failed: %s" % e
        return st["vec"]
    prev, prev_frame = st["prev"], st["frame"]
    st["prev"], st["frame"] = (pix, vis, ok), frame
    if prev is None or prev[0].shape != pix.shape or abs(frame - prev_frame) > 5:
        return st["vec"]
    # Points that are valid both frames and in view in at least one of them
    # (fast film can leave the view between two frames, so do not require both).
    sel = ok & prev[2] & (vis | prev[1])
    if sel.sum() < 8:
        st["text"] = "Tracked points not near the camera view (%d of %d visible)" % (
            int(vis.sum()), len(vis))
        return st["vec"]
    d = (pix - prev[0])[sel]
    a, b, c = (d[:, 0] ** 2).sum(), (d[:, 0] * d[:, 1]).sum(), (d[:, 1] ** 2).sum()
    tr = a + c
    if tr <= 0 or math.sqrt(tr / len(d)) < MIN_MOTION_PX:
        return st["vec"]                                   # not moving: hold last direction
    m = np.array([[a, b], [b, c]]) / tr
    st["M"] = m if st["M"] is None else 0.5 * m + 0.5 * st["M"]
    vals, vecs = np.linalg.eigh(st["M"])
    ux, uy = float(vecs[0, 1]), float(vecs[1, 1])
    if abs(ux) < SNAP:
        ux, uy = 0.0, 1.0
    elif abs(uy) < SNAP:
        ux, uy = 1.0, 0.0
    n = math.hypot(ux, uy)
    st["vec"] = (ux / n, uy / n)
    st["text"] = "Detected: " + _describe(*st["vec"])
    return st["vec"]


def _active_settings(scene):
    cam = scene.camera
    if cam is None or cam.type != 'CAMERA':
        return None
    return getattr(cam.data, "film_blur", None)


def sync(scene):
    """Push the active camera's film-blur settings into the compositor node."""
    s = _active_settings(scene)
    tree = _get_tree(scene)
    node = tree.nodes.get(NODE_NAME) if tree else None
    if s is None or not s.enabled:
        if node is not None and not node.mute:
            node.mute = True
        return
    if node is None:
        node = _ensure_node(scene)
        if node is None:
            return
    scale = scene.render.resolution_percentage / 100.0
    res_x = scene.render.resolution_x * scale
    res_y = scene.render.resolution_y * scale
    L = blur_fraction(s)
    # Box of full width L*dimension -> half-width (kernel radius) in pixels.
    # (In Automatic mode the dimension is taken along the detected direction.)
    vec = None
    if s.direction_mode == 'AUTO':
        # Purely motion-driven. Until the first frame-to-frame motion has been seen
        # there is nothing to detect yet, so start along the vertical.
        vec = _auto_vector(scene, s) or (0.0, 1.0)
    if vec is not None:
        ux, uy = vec
        Lpx = L * math.hypot(ux * res_x, uy * res_y)       # L% of the image along the path
        sx, sy = 0.5 * Lpx * abs(ux), 0.5 * Lpx * abs(uy)
    elif s.axis == 'VERTICAL':
        sx, sy = 0.0, 0.5 * L * res_y
    elif s.axis == 'HORIZONTAL':
        sx, sy = 0.5 * L * res_x, 0.0
    else:                                                 # BOTH: smear along X and Y
        sx, sy = 0.5 * L * res_x, 0.5 * L * res_y
    sx, sy = min(sx, 2048.0), min(sy, 2048.0)
    active = (sx > 0.5 or sy > 0.5)
    _set_box_filter(node)
    if node.mute == active:
        node.mute = not active
    if active:
        _set_blur_size(node, sx, sy)


def _remove_node(scene):
    tree = _get_tree(scene)
    node = tree.nodes.get(NODE_NAME) if tree else None
    if node is None:
        return
    if node.inputs[0].links and node.outputs[0].links:
        src = node.inputs[0].links[0].from_socket
        for l in list(node.outputs[0].links):
            tree.links.new(src, l.to_socket)
    tree.nodes.remove(node)


# ----------------------------------------------------------------------------
# Properties
# ----------------------------------------------------------------------------
def _changed(self, context):
    _AUTO["prev"] = None
    _AUTO["frame"] = None
    try:
        if self.enabled:
            _ensure_node(context.scene)
        sync(context.scene)
    except Exception as e:
        print("Film Motion Blur:", e)


class FilmBlurSettings(PropertyGroup):
    enabled: BoolProperty(
        name="Film Motion Blur", default=False, update=_changed,
        description="Blur what this camera sees as film running through a "
                    "projector would appear (applied via the compositor)")
    film_speed: FloatProperty(
        name="Film Speed", default=24.0, min=0.0, soft_max=96.0, precision=2,
        update=_changed,
        description="Frames of film passing the picture gate per second. "
                    "0 = film stopped (no blur). Animate this to ramp the "
                    "projector up and down")
    integration_ms: FloatProperty(
        name="Eye Integration", default=40.0, min=1.0, soft_max=100.0,
        precision=1, update=_changed,
        description="Time (milliseconds) over which the viewer's eye averages "
                    "light. About 40 ms (one frame at 24 fps) looks natural; "
                    "raise for stronger blur, lower for crisper")
    frame_size: FloatProperty(
        name="Frame Height in View", default=100.0, min=1.0, soft_max=300.0,
        subtype='PERCENTAGE', update=_changed,
        description="How tall one film frame appears in the camera image, as a "
                    "percentage of the image dimension along the travel "
                    "direction (100% = one frame fills the view)")
    direction_mode: EnumProperty(
        name="Direction Mode", update=_changed,
        items=[('MANUAL', "Manual", "Choose the travel direction yourself"),
               ('AUTO', "Automatic", "Detect the travel direction by tracking an "
                                     "object's motion in the camera view")],
        default='MANUAL')
    auto_target: PointerProperty(
        name="Track Object", type=bpy.types.Object, update=_changed,
        description="Object being motion blurred (for example the film strip). Its "
                    "motion frame to frame sets the blur direction. Needs the frame "
                    "to change, so it works when scrubbing, playing or rendering "
                    "an animation")
    auto_collection: PointerProperty(
        name="Track Collection", type=bpy.types.Collection, update=_changed,
        description="Track every object in this collection (use instead of, or as "
                    "well as, a single object)")
    auto_lens: PointerProperty(
        name="Lens or Light", type=bpy.types.Object, update=_changed,
        description="Optional. The lens or lamp the picture is projected from. With a "
                    "Screen set too, each tracked point is followed along the line "
                    "through this object onto the screen. Leave empty to drop the "
                    "points straight onto the screen instead")
    auto_screen: PointerProperty(
        name="Screen", type=bpy.types.Object, update=_changed,
        description="The plane the projector's picture lands on. Set this when the "
                    "film is behind or outside the camera view")
    axis: EnumProperty(
        name="Travel Direction", update=_changed,
        items=[('VERTICAL', "Vertical", "Film moves up/down in the camera image"),
               ('HORIZONTAL', "Horizontal", "Film moves left/right in the camera image"),
               ('BOTH', "Both", "Blur both horizontally and vertically")],
        default='VERTICAL')


# ----------------------------------------------------------------------------
# UI
# ----------------------------------------------------------------------------
class FILMBLUR_OT_remove(Operator):
    bl_idname = "film_blur.remove_node"
    bl_label = "Remove Compositor Node"
    bl_description = "Remove the blur node from the compositor and reconnect the chain"

    def execute(self, context):
        _remove_node(context.scene)
        return {'FINISHED'}


SETTING_NAMES = ("film_speed", "integration_ms", "frame_size", "axis", "direction_mode")


class FILMBLUR_OT_reset(Operator):
    bl_idname = "film_blur.reset"
    bl_label = "Reset Settings"
    bl_description = ("Reset Film Speed, Eye Integration, Frame Height and Travel "
                      "Direction to their defaults (Enable checkbox and tracking objects are kept)")

    @classmethod
    def poll(cls, context):
        return context.camera is not None

    def execute(self, context):
        cam = context.camera
        s = cam.film_blur
        for name in SETTING_NAMES:
            s.property_unset(name)
        reset_auto()
        keyed = False
        try:
            ad = cam.animation_data
            if ad and ad.action:
                keyed = any(fc.data_path.startswith("film_blur.")
                            for fc in ad.action.fcurves)
        except Exception:
            pass
        try:
            sync(context.scene)
        except Exception as e:
            print("Film Motion Blur:", e)
        if keyed:
            self.report({'INFO'}, "Settings reset. Existing keyframes were left in "
                                  "place and will still drive the values.")
        return {'FINISHED'}


class FILMBLUR_OT_refresh(Operator):
    bl_idname = "film_blur.refresh"
    bl_label = "Re-apply to Scene"
    bl_description = "Rebuild the compositor node (placed after Denoise) and update it from this camera"

    def execute(self, context):
        try:
            _remove_node(context.scene)
            _ensure_node(context.scene)
            sync(context.scene)
        except Exception as e:
            self.report({'ERROR'}, f"Could not set up compositor: {e}")
            return {'CANCELLED'}
        return {'FINISHED'}


class FILMBLUR_PT_panel(Panel):
    bl_label = "Film Motion Blur"
    bl_space_type = 'PROPERTIES'
    bl_region_type = 'WINDOW'
    bl_context = "data"
    bl_options = {'DEFAULT_CLOSED'}

    @classmethod
    def poll(cls, context):
        return context.camera is not None

    def draw_header(self, context):
        self.layout.prop(context.camera.film_blur, "enabled", text="")

    def draw(self, context):
        layout = self.layout
        try:
            self._draw(context, layout)
        except Exception as e:                      # show the cause instead of an empty panel
            import traceback
            traceback.print_exc()
            layout.label(text=f"Panel error: {type(e).__name__}: {e}", icon='ERROR')

    def _draw(self, context, layout):
        s = context.camera.film_blur
        layout.use_property_split = True
        col = layout.column()
        col.active = s.enabled
        col.prop(s, "film_speed")
        col.prop(s, "integration_ms")
        col.prop(s, "frame_size")
        col.prop(s, "direction_mode", expand=True)
        if s.direction_mode == 'AUTO':
            col.prop(s, "auto_target", text="Track Object")
            col.prop(s, "auto_collection", text="Track Collection")
            try:
                head, body = col.panel("film_blur_projection", default_closed=False)
                head.label(text="Film behind the camera?")
            except Exception:                              # older Blender: no sub-panels
                head, body = None, col.column()
                body.label(text="Film behind the camera?")
            if body is not None:
                body.prop(s, "auto_screen", text="Screen")
                body.prop(s, "auto_lens", text="Lens or Light")
                if s.auto_lens is not None and s.auto_screen is None:
                    body.label(text="Set a Screen to use the lens or light", icon='ERROR')
            col.label(text=_AUTO["text"], icon='TRACKING')
            axis_name = "the detected direction"
        else:
            col.prop(s, "axis", text="Direction")
            axis_name = {"VERTICAL": "height", "HORIZONTAL": "width",
                         "BOTH": "height and width"}[s.axis]
        L = blur_fraction(s)
        col.label(text="Blur length: %d%% of %s" % (round(L * 100), axis_name))
        row = col.row(align=True)
        row.operator("film_blur.refresh", icon='FILE_REFRESH')
        row.operator("film_blur.remove_node", icon='X', text="Remove")
        col.operator("film_blur.reset", icon='LOOP_BACK')


# ----------------------------------------------------------------------------
# Handlers (keep the node in step with animated camera values)
# ----------------------------------------------------------------------------
@persistent
def _on_frame(scene, depsgraph=None):
    try:
        sync(scene)
    except Exception as e:
        print("Film Motion Blur:", e)


@persistent
def _on_load(*args):
    for sc in bpy.data.scenes:
        _on_frame(sc)


_HANDLERS = (("frame_change_post", _on_frame), ("render_pre", _on_frame),
             ("render_init", _on_frame), ("load_post", _on_load))

classes = (FilmBlurSettings, FILMBLUR_OT_remove, FILMBLUR_OT_reset,
           FILMBLUR_OT_refresh, FILMBLUR_PT_panel)


def register():
    for c in classes:
        bpy.utils.register_class(c)
    bpy.types.Camera.film_blur = PointerProperty(type=FilmBlurSettings)
    for name, fn in _HANDLERS:
        lst = getattr(bpy.app.handlers, name)
        if fn not in lst:
            lst.append(fn)


def unregister():
    for name, fn in _HANDLERS:
        lst = getattr(bpy.app.handlers, name)
        if fn in lst:
            lst.remove(fn)
    del bpy.types.Camera.film_blur
    for c in reversed(classes):
        bpy.utils.unregister_class(c)


if __name__ == "__main__":
    register()
