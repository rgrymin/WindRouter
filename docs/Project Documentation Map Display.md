# **Scampi Navigation \- Multi-Route Monitor**

A specialized Python-based desktop application designed for maritime navigation analysis. It provides real-time monitoring and visualization of multiple GPX tracks, environmental danger zones, and routing graphs using a modern map interface.

## **Key Features**

* **Multi-Route Monitoring**: Real-time tracking of up to 8 independent GPX files:  
  * **4 VMG Routes**: Optimized for Velocity Made Good (displayed in blue shades).  
  * **4 Fastest Path Routes**: Optimized for speed (displayed in red shades).  
* **Environmental Data Overlay**:  
  * **Forbidden Areas**: Highlights high-risk zones based on wind or safety criteria (red rectangles).  
  * **Not Recommended Areas**: Highlights cautionary zones (orange rectangles).  
* **Dynamic Land Masking**: Automatically generates a grey mask over land areas within the operational bounding box to enhance maritime focus.  
* **Routing Graph Visualization**: Visualizes the underlying sailing connection graph (nodes and edges) from JSON data.  
* **Live File Monitoring**: Automatically detects changes in .gpx and .json files every 10 seconds and updates the map view without a restart.  
* **Integrated Map Widget**: Supports standard OpenStreetMap, Satellite imagery, and OpenSeaMap tiles.

## **Prerequisites**

Before running the application, ensure you have Python 3.x installed. You will also need the following libraries:

pip install tkintermapview gpxpy global-land-mask

*Note: tkinter is usually included with standard Python installations.*

## **File Structure**

The application automatically monitors the following files in the working directory:

| Filename | Description |
| :---- | :---- |
| scampi\_vmg\_start\_1..4.gpx | VMG optimized track files. |
| fastest\_path\_start\_1..4.gpx | Fastest path track files. |
| forbidden\_areas.gpx | Waypoints marking danger zones. |
| not\_recommended.gpx | Waypoints marking cautionary zones. |
| sailing\_graph.json | JSON file containing navigation nodes and edges. |

## **Usage**

1. **Launch the Application**: Run the script using Python:  
   python gpx\_viewer.py

2. **Navigation**: Use the mouse to drag the map and the scroll wheel to zoom.  
3. **Controls**:  
   * **Load other GPX**: Manually import a track file to replace the primary VMG route.  
   * **Filter land in areas**: Toggle to hide danger markers that overlap with land.  
   * **Show land mask**: Toggle the grey land overlay.  
   * **Show graph grid**: Toggle the green connection grid visualization.  
   * **Map Type**: Switch between Standard, Satellite, and OpenSeaMap via the dropdown menu.

## **Technical Details**

* **Mapping**: Powered by TkinterMapView.  
* **GPX Parsing**: Handled by gpxpy.  
* **Land Masking**: Utilizes global-land-mask for fast coordinate-based land/water detection.  
* **Auto-Update**: Implemented using Tkinter's after() loop to check file mtime (modification time) every 10,000ms.

*Developed for maritime routing analysis and visualization.*