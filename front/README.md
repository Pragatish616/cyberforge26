# NexBot Robot Character 3D Viewer

This project provides a simple 3D viewer for the NexBot robot character model using Three.js.

## Files

- `index.html` - Main HTML file with embedded JavaScript to display the 3D model
- `nexbot_robot_character_concept.gltf` - The 3D model file in GLTF format
- `nexbot_robot_character_concept.spline` - Original Spline design file (not used in this viewer)

## How to View

### Option 1: Direct Opening (Simple)
1. Simply open `index.html` in a web browser (Chrome, Firefox, Safari, or Edge)
2. The 3D model will load and be displayed in the browser
3. Use mouse controls to interact with the model:
   - Left-click and drag: Rotate the camera
   - Right-click and drag: Pan the camera
   - Scroll wheel: Zoom in/out

### Option 2: Local Server (Recommended for best compatibility)
Some browsers have security restrictions that prevent loading local files via XMLHttpRequest/fetch. If you encounter issues, use a local server:

#### Using Python (if available):
```bash
# Navigate to the project directory
cd /path/to/this/folder

# Start a simple HTTP server
python -m http.server 8000

# Then open in browser:
# http://localhost:8000
```

#### Using Node.js (if available):
```bash
# Install http-server if you don't have it
npm install -g http-server

# Start the server
http-server

# Then open in browser:
# http://localhost:8080
```

## Technical Details

- Uses Three.js r128 for 3D rendering
- Utilizes GLTFLoader to load the .gltf model
- Implements OrbitControls for camera manipulation
- Includes basic lighting (ambient + directional)
- Features a sky-blue background and grid for reference
- Fully responsive - works on desktop and mobile browsers

## Model Features

The NexBot robot character model includes:
- Multiple geometric shapes (cylinders, boxes, ellipses, rectangles)
- Hierarchical structure with various body parts
- Material properties defined in the GLTF file
- Textures and colors as specified in the original design

## Notes

- The model may take a moment to load depending on your internet connection and device performance
- Initial loading shows a progress indicator
- Once loaded, you can interact with the model in real-time
- The camera is automatically positioned to show the entire model when loaded