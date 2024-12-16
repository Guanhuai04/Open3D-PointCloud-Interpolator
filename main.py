import glob
import threading

import numpy as np
import open3d as o3d
import open3d.visualization.gui as gui
import open3d.visualization.rendering as rendering
from scipy.spatial import ConvexHull
from matplotlib.path import Path as MatplotlibPath
import os
import platform
import sys

isMacOS = (platform.system() == "Darwin")

def gram_schmidt(normal, u, v):
    normal = normal / np.linalg.norm(normal)

    u = u - np.dot(u, normal) * normal
    u = u / np.linalg.norm(u)

    v = np.cross(normal, u)
    return normal, u, v

def clone_pcd(pcd):
    new_pcd = o3d.geometry.PointCloud()
    new_pcd.points = o3d.utility.Vector3dVector(np.array(pcd.points))
    new_pcd.colors = o3d.utility.Vector3dVector(np.array(pcd.colors))
    return new_pcd

class Settings:
    def __init__(self):
        self.mouse_model = gui.SceneWidget.Controls.ROTATE_CAMERA
        self.bg_color = gui.Color(1, 1, 1)
        self.show_axes = False
        self.pcd_material = rendering.MaterialRecord()
        self.pcd_material.base_color = [0.0, 0.0, 0.0, 0.5]
        self.pcd_material.point_size = 2
        self.interp_material = rendering.MaterialRecord()
        self.interp_material.base_color = [1.0, 0.0, 0.0, 1.0]
        self.interp_material.point_size = 2
        self.grid_size = 5 # cm
        self.apply_material = True
        self.max_history = 10


class PCInterApp:
    MENU_OPEN = 1
    MENU_SAVE = 2
    MENU_QUIT = 3
    MENU_REDO = 4
    MENU_UNDO = 5
    MENU_INTER = 5
    MENU_CROP = 6
    MENU_SHOW_SETTINGS = 11
    MENU_ABOUT = 21

    def __init__(self, width, height):
        self.settings = Settings()
        self._pcd = None
        self._interp_params = None
        self._interp_result = None
        self._left_ctrl_pressed = False
        self._undo_list = []
        self._redo_list = []

        self.window = gui.Application.instance.create_window(
            "Point Cloud Interpolation", width, height)
        w = self.window  # to make the code more concise

        # 3D widget
        self._scene = gui.SceneWidget()
        self._scene.scene = rendering.Open3DScene(w.renderer)

        em = w.theme.font_size
        separation_height = int(round(0.5 * em))

        self._settings_panel = gui.Vert(
            0, gui.Margins(0.25 * em, 0.25 * em, 0.25 * em, 0.25 * em))

        view_ctrls = gui.CollapsableVert("View controls", 0.25 * em,
                                         gui.Margins(em, 0, 0, 0))

        self._show_axes = gui.Checkbox("Show axes")
        self._show_axes.set_on_checked(self._on_show_axes)
        view_ctrls.add_child(self._show_axes)

        self._settings_panel.add_child(view_ctrls)
        self._settings_panel.add_fixed(separation_height)

        material_settings = gui.CollapsableVert("Material settings", 0,
                                                gui.Margins(em, 0, 0, 0))
        self._point_size = gui.Slider(gui.Slider.INT)
        self._point_size.set_limits(1, 10)
        self._point_size.set_on_value_changed(self._on_point_size)

        point_size_grid = gui.VGrid(2, 0.25 * em)
        point_size_grid.add_child(gui.Label("Point size"))
        point_size_grid.add_child(self._point_size)
        material_settings.add_child(point_size_grid)

        interpolation_settings = gui.CollapsableVert("Interpolation settings", 0,
                                                gui.Margins(em, 0, 0, 0))
        self._grid_size = gui.NumberEdit(gui.NumberEdit.DOUBLE)
        self._grid_size.double_value = 5
        self._grid_size.set_limits(0.1, 100)
        self._grid_size.set_on_value_changed(self._on_grid_size)
        grid_size_grid = gui.VGrid(2, 0.25 * em)
        grid_size_grid.add_child(gui.Label("Grid size(cm)"))
        grid_size_grid.add_child(self._grid_size)
        interpolation_settings.add_child(grid_size_grid)

        hz = gui.Horiz(spacing=10)
        accept_btn = gui.Button('Accept')
        accept_btn.vertical_padding_em = 0
        accept_btn.set_on_clicked(self._on_btn_accept)
        discard_btn = gui.Button('Discard')
        discard_btn.vertical_padding_em = 0
        discard_btn.set_on_clicked(self._on_btn_discard)
        hz.add_child(accept_btn)
        hz.add_child(discard_btn)
        interpolation_settings.add_fixed(separation_height)
        interpolation_settings.add_child(hz)
        interpolation_settings.add_fixed(separation_height)

        self._settings_panel.add_fixed(separation_height)
        self._settings_panel.add_child(material_settings)
        self._settings_panel.add_fixed(separation_height)
        self._settings_panel.add_child(interpolation_settings)

        w.set_on_layout(self._on_layout)
        w.add_child(self._scene)
        w.add_child(self._settings_panel)

        # ---- Menu ----
        # The menu is global (because the macOS menu is global), so only create
        # it once, no matter how many windows are created
        if gui.Application.instance.menubar is None:
            if isMacOS:
                app_menu = gui.Menu()
                app_menu.add_item("About", PCInterApp.MENU_ABOUT)
                app_menu.add_separator()
                app_menu.add_item("Quit(Q)", PCInterApp.MENU_QUIT)
            file_menu = gui.Menu()
            file_menu.add_item("Open...(O)", PCInterApp.MENU_OPEN)
            file_menu.add_item("Save...(Ctrl+S)", PCInterApp.MENU_SAVE)
            if not isMacOS:
                file_menu.add_separator()
                file_menu.add_item("Quit(Q)", PCInterApp.MENU_QUIT)
            edit_menu = gui.Menu()
            edit_menu.add_item("Undo(Ctrl+Z)", PCInterApp.MENU_UNDO)
            edit_menu.add_item("Redo(Ctrl+Y)", PCInterApp.MENU_REDO)
            tools_menu = gui.Menu()
            tools_menu.add_item("Interpolate(I)", PCInterApp.MENU_INTER)
            tools_menu.add_item("Crop(C)", PCInterApp.MENU_CROP)
            settings_menu = gui.Menu()
            settings_menu.add_item("View & Materials",
                                   PCInterApp.MENU_SHOW_SETTINGS)
            settings_menu.set_checked(PCInterApp.MENU_SHOW_SETTINGS, True)
            help_menu = gui.Menu()
            help_menu.add_item("About", PCInterApp.MENU_ABOUT)

            menu = gui.Menu()
            if isMacOS:
                # macOS will name the first menu item for the running application
                # (in our case, probably "Python"), regardless of what we call
                # it. This is the application menu, and it is where the
                # About..., Preferences..., and Quit menu items typically go.
                menu.add_menu("Example", app_menu)
                menu.add_menu("File", file_menu)
                menu.add_menu("Edit", edit_menu)
                menu.add_menu("Tools", tools_menu)
                menu.add_menu("Settings", settings_menu)
                # Don't include help menu unless it has something more than
                # About...
            else:
                menu.add_menu("File", file_menu)
                menu.add_menu("Tools", tools_menu)
                menu.add_menu("Edit", edit_menu)
                menu.add_menu("Settings", settings_menu)
                menu.add_menu("Help", help_menu)
            gui.Application.instance.menubar = menu

        # The menubar is global, but we need to connect the menu items to the
        # window, so that the window can call the appropriate function when the
        # menu item is activated.
        w.set_on_menu_item_activated(PCInterApp.MENU_OPEN, self._on_menu_open)
        w.set_on_menu_item_activated(PCInterApp.MENU_SAVE,
                                     self._on_menu_save)
        w.set_on_menu_item_activated(PCInterApp.MENU_QUIT, self._on_menu_quit)
        w.set_on_menu_item_activated(PCInterApp.MENU_UNDO, self._on_undo())
        w.set_on_menu_item_activated(PCInterApp.MENU_REDO, self._on_redo())
        w.set_on_menu_item_activated(PCInterApp.MENU_INTER, self._on_menu_interp)
        w.set_on_menu_item_activated(PCInterApp.MENU_CROP, self._on_menu_crop)
        w.set_on_menu_item_activated(PCInterApp.MENU_SHOW_SETTINGS,
                                     self._on_menu_toggle_settings_panel)
        w.set_on_menu_item_activated(PCInterApp.MENU_ABOUT, self._on_menu_about)
        w.set_on_key(self._on_key_callback)
        # ----

        self._apply_settings()


    def _apply_settings(self):
        bg_color = [
            self.settings.bg_color.red, self.settings.bg_color.green,
            self.settings.bg_color.blue, self.settings.bg_color.alpha
        ]
        self._scene.scene.set_background(bg_color)
        self._scene.scene.show_axes(self.settings.show_axes)

        if self.settings.apply_material:
            if self._scene.scene.has_geometry("__pcd__"):
                self._scene.scene.modify_geometry_material("__pcd__", self.settings.pcd_material)
            if self._scene.scene.has_geometry("__interp__"):
                self._scene.scene.modify_geometry_material("__interp__", self.settings.interp_material)
            self.settings.apply_material = False

        self._show_axes.checked = self.settings.show_axes
        self._point_size.double_value = self.settings.pcd_material.point_size

    def _on_layout(self, layout_context):
        # The on_layout callback should set the frame (position + size) of every
        # child correctly. After the callback is done the window will layout
        # the grandchildren.
        r = self.window.content_rect
        self._scene.frame = r
        width = 17 * layout_context.theme.font_size
        height = min(
            r.height,
            self._settings_panel.calc_preferred_size(
                layout_context, gui.Widget.Constraints()).height)
        self._settings_panel.frame = gui.Rect(r.get_right() - width, r.y, width, height)

    def _set_mouse_mode_rotate(self):
        self._scene.set_view_controls(gui.SceneWidget.Controls.ROTATE_CAMERA)

    def _set_mouse_mode_model(self):
        self._scene.set_view_controls(gui.SceneWidget.Controls.ROTATE_MODEL)

    def _on_show_axes(self, show):
        self.settings.show_axes = show
        self._apply_settings()

    def _on_point_size(self, size):
        self.settings.pcd_material.point_size = int(size)
        self.settings.interp_material.point_size = int(size)
        self.settings.apply_material = True
        self._apply_settings()

    def _on_grid_size(self, size):
        self.settings.grid_size = size
        if self._interp_params is not None:
            interp_points = self._interpolate_region(self._interp_params["region_pcd"])
            interp_pcd = self._get_select_region(interp_points,
                                                 np.array(self._interp_params["region_pcd"].points),
                                                 self._interp_params["region_cam_extrinsic"])
            selected_pcd = self._get_select_region(np.array(interp_pcd.points),
                                                   np.array(self._interp_params["selected_pcd"].points),
                                                   self._interp_params["selected_cam_extrinsic"])
            if self._scene.scene.has_geometry("__interp__"):
                self._scene.scene.remove_geometry("__interp__")
            self._scene.scene.add_geometry("__interp__", selected_pcd, self.settings.interp_material)
            self._interp_result = selected_pcd

    def _on_btn_accept(self):
        if self._interp_params is not None:
            if len(self._undo_list) >= self.settings.max_history:
                self._undo_list.pop(0)
            self._undo_list.append(clone_pcd(self._pcd))
            self._redo_list.clear()
            self._pcd = self._pcd + self._interp_result
            self._scene.scene.remove_geometry("__interp__")
            self._scene.scene.remove_geometry("__pcd__")
            self._scene.scene.add_geometry("__pcd__", self._pcd, self.settings.pcd_material)
            self._interp_params = None
            self._interp_result = None
            self._apply_settings()

    def _on_btn_discard(self):
        if self._interp_params is not None:
            self._scene.scene.remove_geometry("__interp__")
            self._interp_params = None
            self._interp_result = None

    def _on_undo(self):
        if len(self._undo_list) == 0:
            return
        self._scene.scene.remove_geometry("__pcd__")
        self._redo_list.append(clone_pcd(self._pcd))
        self._pcd = self._undo_list.pop()
        self._scene.scene.add_geometry("__pcd__", self._pcd, self.settings.pcd_material)

    def _on_redo(self):
        if len(self._redo_list) == 0:
            return
        self._scene.scene.remove_geometry("__pcd__")
        self._undo_list.append(clone_pcd(self._pcd))
        self._pcd = self._redo_list.pop()
        self._scene.scene.add_geometry("__pcd__", self._pcd, self.settings.pcd_material)

    def _on_key_callback(self, event):
        if event.type == gui.KeyEvent.Type.UP:
            if event.key == gui.KeyName.LEFT_CONTROL:
                self._left_ctrl_pressed = False
            elif event.key == gui.KeyName.O:
                self._on_menu_open()
            elif event.key == gui.KeyName.S:
                if self._left_ctrl_pressed:
                    self._on_menu_save()
            elif event.key == gui.KeyName.Q:
                self._on_menu_quit()
            elif event.key == gui.KeyName.I:
                self._on_menu_interp()
            elif event.key == gui.KeyName.C:
                self._on_menu_crop()
            elif event.key == gui.KeyName.A:
                self._on_btn_accept()
            elif event.key == gui.KeyName.D:
                self._on_btn_discard()
            elif event.key == gui.KeyName.Z:
                if self._left_ctrl_pressed:
                    self._on_undo()
            elif event.key == gui.KeyName.Y:
                if self._left_ctrl_pressed:
                    self._on_redo()
        if event.type == gui.KeyEvent.Type.DOWN:
            if event.key == gui.KeyName.LEFT_CONTROL:
                self._left_ctrl_pressed = True
        return True

    def _on_menu_open(self):
        dlg = gui.FileDialog(gui.FileDialog.OPEN, "Choose file to load",
                             self.window.theme)
        dlg.add_filter(
            ".xyz .xyzn .xyzrgb .ply .pcd .pts",
            "Point cloud files (.xyz, .xyzn, .xyzrgb, .ply, "
            ".pcd, .pts)")
        dlg.add_filter("", "All files")

        # A file dialog MUST define on_cancel and on_done functions
        dlg.set_on_cancel(self._on_file_dialog_cancel)
        dlg.set_on_done(self._on_load_dialog_done)
        self.window.show_dialog(dlg)

    def _on_menu_interp(self):
        if self._scene.scene.has_geometry("__interp__"):
            self._scene.scene.remove_geometry("__interp__")
        self._interp_params = None
        self._interp_result = None
        if self._pcd is None:
            self._show_error_dialog("Interpolation failed", "Please load a pcd file first.")
            return

        # 选择待插值区域
        region_pcd, region_cam_extrinsic = self._select_region("Select region for interpolation",
                                                        self._pcd.paint_uniform_color([0, 0, 0]))
        print("view_extrinsic:", region_cam_extrinsic)

        # 获取选中点的坐标
        if (region_pcd is None or len(region_pcd.points) < 3 or
                len(region_pcd.points) == len(self._pcd.points)):
            self._show_error_dialog("Interpolation failed",
                                    "Select region not detected, please press 'c' after selecting the region!")
            return

        # 执行插值
        interp_points = self._interpolate_region(region_pcd)
        if interp_points is None:
            self._show_error_dialog("Interpolation failed", "Too few points selected.")
            return
        interp_pcd = self._get_select_region(interp_points, np.array(region_pcd.points), region_cam_extrinsic)

        # 选择需要保存的区域
        selected_region, selected_cam_extrinsic = self._select_region("Select point cloud to keep",
                                                             interp_pcd.paint_uniform_color(
                                                                 [1, 0, 0]) + self._pcd.paint_uniform_color(
                                                                 [0, 0, 0]))
        print("view_extrinsic:", selected_cam_extrinsic)
        if (selected_region is None or len(selected_region.points) == 0 or
                len(selected_region.points) == len(self._pcd.points) + len(interp_pcd.points)):
            self._show_error_dialog("Interpolation failed", "Please select a region to keep.")
            return
        self._interp_params = dict(
            region_pcd=region_pcd,
            region_cam_extrinsic=region_cam_extrinsic,
            selected_pcd=selected_region,
            selected_cam_extrinsic=selected_cam_extrinsic
        )

        selected_pcd = self._get_select_region(np.array(interp_pcd.points), np.array(selected_region.points),
                                               selected_cam_extrinsic)
        self._scene.scene.add_geometry("__interp__", selected_pcd, self.settings.interp_material)
        self._interp_result = selected_pcd

    def _on_menu_crop(self):
        if self._pcd is None:
            self._show_error_dialog("Interpolation failed", "Please load a pcd file first.")
            return
        # 选择待裁剪区域
        region_pcd, cam_extrinsic = self._select_region("Select region for interpolation",
                                                        self._pcd.paint_uniform_color([0, 0, 0]))
        print("view_extrinsic:", cam_extrinsic)

        if (region_pcd is None or len(region_pcd.points) < 3 or
                len(region_pcd.points) == len(self._pcd.points)):
            self._show_error_dialog("Crop failed",
                                    "Select region not detected, please press 'c' after selecting the region!")
            return

        cropped_pcd = self._crop_pcd(self._pcd, region_pcd, cam_extrinsic)
        self._undo_list.append(clone_pcd(self._pcd))
        self._redo_list.clear()
        self._scene.scene.remove_geometry("__pcd__")
        self._pcd = cropped_pcd
        self._scene.scene.add_geometry("__pcd__", self._pcd, self.settings.pcd_material)

    def _on_file_dialog_cancel(self):
        self.window.close_dialog()

    def _on_load_dialog_done(self, filename):
        self.window.close_dialog()
        self.load(filename)

    def _show_error_dialog(self, title, message):
        em = self.window.theme.font_size
        dlg = gui.Dialog(title)

        # Add the text
        dlg_layout = gui.Vert(em, gui.Margins(em, em, em, em))
        dlg_layout.add_child(gui.Label(message))

        # Add the Ok button. We need to define a callback function to handle
        # the click.
        ok = gui.Button("OK")
        ok.set_on_clicked(self._on_about_ok)

        # We want the Ok button to be an the right side, so we need to add
        # a stretch item to the layout, otherwise the button will be the size
        # of the entire row. A stretch item takes up as much space as it can,
        # which forces the button to be its minimum size.
        h = gui.Horiz()
        h.add_stretch()
        h.add_child(ok)
        h.add_stretch()
        dlg_layout.add_child(h)

        dlg.add_child(dlg_layout)
        self.window.show_dialog(dlg)

    def _on_menu_save(self):
        if self._pcd is None:
            self._show_error_dialog("Save pcd failed", "Please load a pcd file first.")
            return
        dlg = gui.FileDialog(gui.FileDialog.SAVE, "Choose file to save",
                             self.window.theme)
        dlg.add_filter(".pcd", "Point Cloud Data files (.pcd)")
        dlg.set_on_cancel(self._on_file_dialog_cancel)
        dlg.set_on_done(self._on_save_dialog_done)
        self.window.show_dialog(dlg)

    def _on_save_dialog_done(self, filename):
        self.window.close_dialog()
        self.save_pcd(filename)

    def _on_menu_quit(self):
        gui.Application.instance.quit()

    def _on_menu_toggle_settings_panel(self):
        self._settings_panel.visible = not self._settings_panel.visible
        gui.Application.instance.menubar.set_checked(
            PCInterApp.MENU_SHOW_SETTINGS, self._settings_panel.visible)

    def _on_menu_about(self):
        # Show a simple dialog. Although the Dialog is actually a widget, you can
        # treat it similar to a Window for layout and put all the widgets in a
        # layout which you make the only child of the Dialog.
        em = self.window.theme.font_size
        dlg = gui.Dialog("About")

        # Add the text
        dlg_layout = gui.Vert(em, gui.Margins(em, em, em, em))
        dlg_layout.add_child(gui.Label("Open3D PointCloud Interpolation App"))
        dlg_layout.add_child(gui.Label("Author: Guan Huai"))

        # Add the Ok button. We need to define a callback function to handle
        # the click.
        ok = gui.Button("OK")
        ok.set_on_clicked(self._on_about_ok)

        # We want the Ok button to be an the right side, so we need to add
        # a stretch item to the layout, otherwise the button will be the size
        # of the entire row. A stretch item takes up as much space as it can,
        # which forces the button to be its minimum size.
        h = gui.Horiz()
        h.add_stretch()
        h.add_child(ok)
        h.add_stretch()
        dlg_layout.add_child(h)

        dlg.add_child(dlg_layout)
        self.window.show_dialog(dlg)

    def _on_about_ok(self):
        self.window.close_dialog()

    def load(self, path):
        self._scene.scene.clear_geometry()

        geometry = None

        cloud = None
        try:
            cloud = o3d.io.read_point_cloud(path)
        except Exception:
            pass
        if cloud is not None:
            print("[Info] Successfully read", path)
            if not cloud.has_normals():
                cloud.estimate_normals()
            cloud.normalize_normals()
            geometry = cloud
        else:
            print("[WARNING] Failed to read points", path)

        if geometry is not None:
            try:
                # Point cloud
                self._scene.scene.remove_geometry("__pcd__")
                self._scene.scene.add_geometry("__pcd__", geometry,
                                                   self.settings.pcd_material)
                self._pcd = geometry
                bounds = self._scene.scene.bounding_box
                self._scene.setup_camera(60, bounds, bounds.get_center())
            except Exception as e:
                print(e)

    def save_pcd(self, path):
        o3d.io.write_point_cloud(path, self._pcd)

    def export_image(self, path, width, height):

        def on_image(image):
            img = image

            quality = 9  # png
            if path.endswith(".jpg"):
                quality = 100
            o3d.io.write_image(path, img, quality)

        self._scene.scene.scene.render_to_image(on_image)

    def _select_region(self, window_name, pcd):
        vis = o3d.visualization.VisualizerWithEditing()
        vis.create_window(window_name=window_name)
        render_option = vis.get_render_option()
        render_option.point_size = self.settings.pcd_material.point_size
        vis.add_geometry(pcd)
        vis.run()
        vis.destroy_window()

        extrinsic = vis.get_view_control().convert_to_pinhole_camera_parameters().extrinsic
        rot = extrinsic[:3, :3]
        if not np.allclose(rot @ rot.T, np.eye(3), atol=1e-5):
            extrinsic = np.diag([1, -1, -1, 1])
        return vis.get_cropped_geometry(), extrinsic

    def _interpolate_region(self, selected_pcd) -> np.ndarray:
        # 获取选中点的坐标
        points = np.asarray(selected_pcd.points)
        if len(selected_pcd.points) < 3:
            return None

        # 使用PCA找到最佳拟合平面
        mean = np.mean(selected_pcd.points, axis=0)
        centered = selected_pcd.points - mean
        cov = np.cov(centered.T)
        eigenvalues, eigenvectors = np.linalg.eigh(cov)
        normal = eigenvectors[:, 0]  # 最小特征值对应的特征向量作为法向量

        # 获取平面的两个基向量
        normal, u, v = gram_schmidt(normal, eigenvectors[:, 1], eigenvectors[:, 2])

        # 将顶点投影到平面上
        bounding_box = np.array(selected_pcd.get_oriented_bounding_box().get_box_points())
        u_coords = np.dot(bounding_box - mean, u)
        v_coords = np.dot(bounding_box - mean, v)

        # 计算u和v方向的范围
        u_min, u_max = np.min(u_coords), np.max(u_coords)
        v_min, v_max = np.min(v_coords), np.max(v_coords)

        # 在平面上生成网格点
        u_num_points = int(np.ceil((u_max - u_min) / (self.settings.grid_size / 100))) + 1
        v_num_points = int(np.ceil((v_max - v_min) / (self.settings.grid_size / 100))) + 1
        u_range = np.linspace(u_min, u_max, u_num_points)
        v_range = np.linspace(v_min, v_max, v_num_points)
        U, V = np.meshgrid(u_range, v_range)

        # 生成插值点
        interp_points = mean + U.flatten()[:, None] * u + V.flatten()[:, None] * v

        return interp_points

    def _get_select_region(self, interp_points, keep_points, cam_extrinsic) -> o3d.geometry.PointCloud:
        # 将选中的点投影到相机平面
        selected_points_homo = np.hstack([keep_points, np.ones((len(keep_points), 1))])
        proj_selected = (cam_extrinsic @ selected_points_homo.T).T
        proj_selected = proj_selected[:, :2]

        # 计算投影点的凸包
        hull = ConvexHull(proj_selected)
        hull_points = proj_selected[hull.vertices]

        # 将插值点投影到相机平面
        interp_homo = np.hstack([interp_points, np.ones((len(interp_points), 1))])
        proj_interp = (cam_extrinsic @ interp_homo.T).T
        proj_interp = proj_interp[:, :2]

        # 判断插值点是否在凸包内
        hull_path = MatplotlibPath(hull_points)
        valid_idx = hull_path.contains_points(proj_interp)

        # 创建插值点云
        colors = np.zeros((len(interp_points), 3))
        interp_pcd = o3d.geometry.PointCloud()
        interp_pcd.points = o3d.utility.Vector3dVector(interp_points[valid_idx])
        # interp_pcd.colors = o3d.utility.Vector3dVector(colors[valid_idx])

        return interp_pcd

    def _crop_pcd(self, origin_pcd, region_pcd, cam_extrinsic):
        # 将选中的点投影到相机平面
        selected_points_homo = np.hstack([np.array(region_pcd.points), np.ones((len(region_pcd.points), 1))])
        proj_selected = (cam_extrinsic @ selected_points_homo.T).T
        proj_selected = proj_selected[:, :2]

        # 计算投影点的凸包
        hull = ConvexHull(proj_selected)
        hull_points = proj_selected[hull.vertices]

        # 将原始点投影到相机平面
        interp_homo = np.hstack([np.array(origin_pcd.points), np.ones((len(origin_pcd.points), 1))])
        proj_interp = (cam_extrinsic @ interp_homo.T).T
        proj_interp = proj_interp[:, :2]

        # 判断插值点是否在凸包内
        hull_path = MatplotlibPath(hull_points)
        valid_idx = hull_path.contains_points(proj_interp)

        # 创建插值点云
        colors = np.zeros((len(origin_pcd.points), 3))
        interp_pcd = o3d.geometry.PointCloud()
        interp_pcd.points = o3d.utility.Vector3dVector(np.array(origin_pcd.points)[~valid_idx])
        interp_pcd.colors = o3d.utility.Vector3dVector(colors[~valid_idx])

        return interp_pcd

def main():
    # We need to initialize the application, which finds the necessary shaders
    # for rendering and prepares the cross-platform window abstraction.
    gui.Application.instance.initialize()

    w = PCInterApp(1920, 1080)

    if len(sys.argv) > 1:
        path = sys.argv[1]
        if os.path.exists(path):
            w.load(path)
        else:
            w.window.show_message_box("Error",
                                      "Could not open file '" + path + "'")
    vis = o3d.visualization.VisualizerWithEditing() # 解决  GLFW Error: The GLFW library is not initialized
    vis.create_window(visible=False)
    # Run the event loop. This will not return until the last window is closed.
    gui.Application.instance.run()
    vis.destroy_window()


if __name__ == "__main__":
    main()