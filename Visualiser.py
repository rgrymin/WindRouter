import tkinter as tk
from tkinter import filedialog, messagebox
import tkintermapview
import gpxpy
import gpxpy.gpx
import os
import json

# Attempt to import the land check library
try:
    from global_land_mask import globe
    HAS_LAND_MASK = True
except ImportError:
    HAS_LAND_MASK = False

class GPXViewerApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Scampi Navigation - Multi-Route Monitor Full (3D Enabled)")
        self.root.geometry("1550x950")

        # Route file definitions
        self.vmg_files = [f"output/scampi_vmg_start_{i}.gpx" for i in range(1, 5)]
        self.fastest_files = [f"output/fastest_path_start_{i}.gpx" for i in range(1, 5)]
        self.fastest_3d_files = [f"output/fastest_path_3d_start_{i}.gpx" for i in range(1, 5)]

        # Additional files
        self.forbidden_file = "output/forbidden_areas.gpx"
        self.not_recommended_file = "output/not_recommended.gpx"
        self.graph_file = "output/sailing_graph.json"
        
        # Initialize modification times (mtimes)
        self.last_vmg_mtimes = [0] * 4
        self.last_fastest_mtimes = [0] * 4
        self.last_fastest_3d_mtimes = [0] * 4
        self.last_forbidden_mtime = 0
        self.last_not_recommended_mtime = 0
        self.last_graph_mtime = 0

        # Map objects
        self.vmg_objs = [None] * 4
        self.fastest_objs = [None] * 4
        self.fastest_3d_objs = [None] * 4
        self.forbidden_shapes = []
        self.not_recommended_shapes = []
        self.land_mask_shapes = []
        self.graph_segments = []

        # Data boundaries (Bounding Box)
        self.min_lat, self.max_lat = 90, -90
        self.min_lon, self.max_lon = 180, -180

        # Color palette
        self.vmg_colors = ["#1F77B4", "#3498DB", "#5DADE2", "#85C1E9"] # Blue shades
        self.fastest_colors = ["#C0392B", "#E74C3C", "#EC7063", "#F1948A"] # Red shades
        self.fastest_3d_colors = ["#F1C40F", "#F39C12", "#D4AC0D", "#B7950B"] # Yellow/Gold shades

        self.setup_ui()
        self.check_files_loop()

    def setup_ui(self):
        # --- Top control panel ---
        self.top_frame = tk.Frame(self.root)
        self.top_frame.pack(side=tk.TOP, fill=tk.X, padx=10, pady=10)

        # 1. Status panel for routes
        self.status_frame = tk.LabelFrame(self.top_frame, text="Route Status (VMG, Fastest & 3D)")
        self.status_frame.pack(side=tk.LEFT, padx=5)

        self.lbl_vmg = []
        for i in range(4):
            lbl = tk.Label(self.status_frame, text=f"VMG {i+1}: None", fg=self.vmg_colors[i], font=("Arial", 9, "bold"))
            lbl.grid(row=i, column=0, sticky="w", padx=10)
            self.lbl_vmg.append(lbl)

        self.lbl_fastest = []
        for i in range(4):
            lbl = tk.Label(self.status_frame, text=f"Fastest {i+1}: None", fg=self.fastest_colors[i], font=("Arial", 9, "bold"))
            lbl.grid(row=i, column=1, sticky="w", padx=10)
            self.lbl_fastest.append(lbl)

        self.lbl_fastest_3d = []
        for i in range(4):
            lbl = tk.Label(self.status_frame, text=f"3D {i+1}: None", fg=self.fastest_3d_colors[i], font=("Arial", 9, "bold"))
            lbl.grid(row=i, column=2, sticky="w", padx=10)
            self.lbl_fastest_3d.append(lbl)

        # 2. Environmental data panel
        self.extra_frame = tk.LabelFrame(self.top_frame, text="Environmental Data")
        self.extra_frame.pack(side=tk.LEFT, padx=10)

        self.lbl_graph_info = tk.Label(self.extra_frame, text="Graph: Waiting...", fg="green")
        self.lbl_graph_info.pack(anchor="w", padx=5)
        
        self.lbl_forbidden_info = tk.Label(self.extra_frame, text="Forbidden: Waiting...", fg="darkred")
        self.lbl_forbidden_info.pack(anchor="w", padx=5)

        self.lbl_not_recom_info = tk.Label(self.extra_frame, text="Not Recommended: Waiting...", fg="#E67E22")
        self.lbl_not_recom_info.pack(anchor="w", padx=5)

        # 3. Options and switches panel
        self.options_frame = tk.Frame(self.top_frame)
        self.options_frame.pack(side=tk.RIGHT, padx=10)

        self.btn_manual = tk.Button(self.options_frame, text="Load other GPX", command=self.manual_load_gpx)
        self.btn_manual.pack(fill=tk.X, pady=2)

        self.filter_land_var = tk.BooleanVar(value=False)
        tk.Checkbutton(self.options_frame, text="Filter land in areas", variable=self.filter_land_var, command=self.force_reload_areas).pack(anchor="e")

        self.show_land_mask_var = tk.BooleanVar(value=True)
        tk.Checkbutton(self.options_frame, text="Show land mask (Grey)", variable=self.show_land_mask_var, command=self.draw_global_land_mask).pack(anchor="e")

        self.show_graph_var = tk.BooleanVar(value=True)
        tk.Checkbutton(self.options_frame, text="Show graph grid", variable=self.show_graph_var, command=self.load_sailing_graph).pack(anchor="e")

        if not HAS_LAND_MASK:
            self.lbl_forbidden_info.config(text="global-land-mask library missing")

        # 4. Map type selection
        self.map_type_var = tk.StringVar(value="Standard")
        self.map_selector = tk.OptionMenu(self.top_frame, self.map_type_var, "Standard", "Satellite", "OpenSeaMap", command=self.change_map_type)
        self.map_selector.pack(side=tk.RIGHT, padx=10)
        tk.Label(self.top_frame, text="Map:").pack(side=tk.RIGHT)

        # --- Map Widget ---
        self.map_widget = tkintermapview.TkinterMapView(self.root, width=1400, height=750, corner_radius=0)
        self.map_widget.pack(fill=tk.BOTH, expand=True)
        
        self.map_widget.set_position(58.0, 10.0) 
        self.map_widget.set_zoom(6)

    def change_map_type(self, selection):
        if selection == "Satellite":
            self.map_widget.set_tile_server("https://mt0.google.com/vt/lyrs=s&x={x}&y={y}&z={z}")
        elif selection == "OpenSeaMap":
            self.map_widget.set_tile_server("https://tiles.openseamap.org/seamark/{z}/{x}/{y}.png")
        else:
            self.map_widget.set_tile_server("https://a.tile.openstreetmap.org/{z}/{x}/{y}.png")

    def update_bounds(self, lat, lon):
        self.min_lat = min(self.min_lat, lat)
        self.max_lat = max(self.max_lat, lat)
        self.min_lon = min(self.min_lon, lon)
        self.max_lon = max(self.max_lon, lon)

    def is_square_land(self, lat, lon, delta):
        if not HAS_LAND_MASK: return False
        return globe.is_land(lat, lon)

    def draw_global_land_mask(self):
        if not HAS_LAND_MASK or not self.show_land_mask_var.get():
            for shape in self.land_mask_shapes: shape.delete()
            self.land_mask_shapes.clear()
            return
        
        if self.min_lat > self.max_lat: return

        for shape in self.land_mask_shapes: shape.delete()
        self.land_mask_shapes.clear()

        s_lat, n_lat = self.min_lat - 1, self.max_lat + 1
        w_lon, e_lon = self.min_lon - 1, self.max_lon + 1
        
        step, delta = 0.25, 0.125
        lat = s_lat
        while lat <= n_lat:
            lon = w_lon
            while lon <= e_lon:
                if self.is_square_land(lat, lon, delta):
                    corners = [(lat-delta, lon-delta), (lat+delta, lon-delta), (lat+delta, lon+delta), (lat-delta, lon+delta)]
                    poly = self.map_widget.set_polygon(corners, fill_color="#555555", outline_color="#333333", border_width=0)
                    self.land_mask_shapes.append(poly)
                lon += step
            lat += step

    def load_sailing_graph(self):
        if not os.path.exists(self.graph_file) or not self.show_graph_var.get():
            for seg in self.graph_segments: seg.delete()
            self.graph_segments.clear()
            return
        try:
            with open(self.graph_file, 'r', encoding='utf-8') as f:
                data = json.load(f)
            nodes = data.get("nodes", {})
            adj = data.get("graph", {})
            for seg in self.graph_segments: seg.delete()
            self.graph_segments.clear()
            
            edges_drawn = 0
            for s_id, targets in adj.items():
                if s_id not in nodes: continue
                s_coords = (nodes[s_id]["lat"], nodes[s_id]["lon"])
                self.update_bounds(*s_coords)
                for edge in targets:
                    t_id = edge["target"]
                    if t_id in nodes:
                        t_coords = (nodes[t_id]["lat"], nodes[t_id]["lon"])
                        line = self.map_widget.set_path([s_coords, t_coords], color="#2ECC71", width=1)
                        self.graph_segments.append(line)
                        edges_drawn += 1
            self.lbl_graph_info.config(text=f"Graph: {len(nodes)} nodes, {edges_drawn} edges")
        except Exception as e: print(f"Graph Error: {e}")

    def load_track(self, file_path, index, type_key, force=False):
        """
        Loads track and draws path line.
        type_key: 'vmg', 'fastest', or '3d'
        """
        if not os.path.exists(file_path): return
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                gpx = gpxpy.parse(f)
            pts = []
            for trk in gpx.tracks:
                for seg in trk.segments:
                    for p in seg.points:
                        pts.append((p.latitude, p.longitude))
                        self.update_bounds(p.latitude, p.longitude)
            if pts:
                if type_key == 'vmg':
                    if self.vmg_objs[index]: self.vmg_objs[index].delete()
                    self.vmg_objs[index] = self.map_widget.set_path(pts, color=self.vmg_colors[index], width=3)
                    self.lbl_vmg[index].config(text=f"VMG {index+1}: {len(pts)} pts")
                elif type_key == 'fastest':
                    if self.fastest_objs[index]: self.fastest_objs[index].delete()
                    self.fastest_objs[index] = self.map_widget.set_path(pts, color=self.fastest_colors[index], width=3)
                    self.lbl_fastest[index].config(text=f"Fast {index+1}: {len(pts)} pts")
                elif type_key == '3d':
                    if self.fastest_3d_objs[index]: self.fastest_3d_objs[index].delete()
                    self.fastest_3d_objs[index] = self.map_widget.set_path(pts, color=self.fastest_3d_colors[index], width=3)
                    self.lbl_fastest_3d[index].config(text=f"3D {index+1}: {len(pts)} pts")
                
                if force and index == 0:
                    self.map_widget.set_position(pts[0][0], pts[0][1])
        except Exception as e: print(f"File Error {file_path}: {e}")

    def load_area_file(self, file_path, shapes_list, fill_color, outline_color, info_label, prefix):
        if not os.path.exists(file_path): return
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                gpx = gpxpy.parse(f)
            for s in shapes_list: s.delete()
            shapes_list.clear()
            
            delta, count, skipped = 0.12, 0, 0
            for wpt in gpx.waypoints:
                self.update_bounds(wpt.latitude, wpt.longitude)
                if self.filter_land_var.get() and HAS_LAND_MASK:
                    if self.is_square_land(wpt.latitude, wpt.longitude, delta):
                        skipped += 1
                        continue
                corners = [(wpt.latitude-delta, wpt.longitude-delta), (wpt.latitude+delta, wpt.longitude-delta),
                           (wpt.latitude+delta, wpt.longitude+delta), (wpt.latitude-delta, wpt.longitude+delta)]
                shapes_list.append(self.map_widget.set_polygon(corners, fill_color=fill_color, outline_color=outline_color, border_width=1))
                count += 1
            info_label.config(text=f"{prefix}: {count} areas ({skipped} hidden)")
        except Exception as e: print(f"Area Error: {e}")

    def force_reload_areas(self):
        self.last_vmg_mtimes = [0]*4
        self.last_fastest_mtimes = [0]*4
        self.last_fastest_3d_mtimes = [0]*4
        self.last_forbidden_mtime = 0
        self.last_not_recommended_mtime = 0
        self.last_graph_mtime = 0
        self.check_files_loop(force=True)

    def check_files_loop(self, force=False):
        any_update = False
        
        # Monitor VMG routes
        for i, path in enumerate(self.vmg_files):
            if os.path.exists(path):
                m = os.path.getmtime(path)
                if m > self.last_vmg_mtimes[i] or force:
                    self.load_track(path, i, 'vmg', force)
                    self.last_vmg_mtimes[i] = m
                    any_update = True

        # Monitor Fastest routes
        for i, path in enumerate(self.fastest_files):
            if os.path.exists(path):
                m = os.path.getmtime(path)
                if m > self.last_fastest_mtimes[i] or force:
                    self.load_track(path, i, 'fastest', force)
                    self.last_fastest_mtimes[i] = m
                    any_update = True

        # Monitor 3D Fastest routes
        for i, path in enumerate(self.fastest_3d_files):
            if os.path.exists(path):
                m = os.path.getmtime(path)
                if m > self.last_fastest_3d_mtimes[i] or force:
                    self.load_track(path, i, '3d', force)
                    self.last_fastest_3d_mtimes[i] = m
                    any_update = True

        # Graph, Forbidden, Not Recommended (same logic)
        if os.path.exists(self.graph_file):
            m = os.path.getmtime(self.graph_file)
            if m > self.last_graph_mtime or force:
                self.load_sailing_graph(); self.last_graph_mtime = m; any_update = True
        
        if os.path.exists(self.forbidden_file):
            m = os.path.getmtime(self.forbidden_file)
            if m > self.last_forbidden_mtime or force:
                self.load_area_file(self.forbidden_file, self.forbidden_shapes, "#FF0000", "#8B0000", self.lbl_forbidden_info, "Forbidden")
                self.last_forbidden_mtime = m; any_update = True

        if os.path.exists(self.not_recommended_file):
            m = os.path.getmtime(self.not_recommended_file)
            if m > self.last_not_recommended_mtime or force:
                self.load_area_file(self.not_recommended_file, self.not_recommended_shapes, "#FFA500", "#D35400", self.lbl_not_recom_info, "Not Recommended")
                self.last_not_recommended_mtime = m; any_update = True

        if any_update and self.show_land_mask_var.get():
            self.draw_global_land_mask()

        if not force:
            self.root.after(10000, self.check_files_loop)

    def manual_load_gpx(self):
        f = filedialog.askopenfilename(filetypes=[("GPX files", "*.gpx")])
        if f:
            self.vmg_files[0] = f
            self.last_vmg_mtimes[0] = 0
            self.load_track(f, 0, 'vmg', True)

if __name__ == "__main__":
    root = tk.Tk()
    app = GPXViewerApp(root)
    root.mainloop()