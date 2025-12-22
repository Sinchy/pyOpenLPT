import base64
import os

base_path = r"d:\0.Code\OpenLPTGUI\OpenLPT\modules\camera_calibration"
html_path = os.path.join(base_path, "WAND_CALIBRATION_USER_GUIDE.html")

def get_base64_src(filename):
    path = os.path.join(base_path, filename)
    if not os.path.exists(path):
        return None
    with open(path, "rb") as f:
        data = f.read()
        encoded = base64.b64encode(data).decode('utf-8')
        # Check header
        if data.startswith(b'\xff\xd8'):
            mime = "image/jpeg"
        else:
            mime = "image/png"
        return f"data:{mime};base64,{encoded}"

# Load content
with open(html_path, "r", encoding="utf-8") as f:
    html = f.read()

# Replace Step 2 image (already embedded but potentially wrong mime)
old_img_tag_start = '<img src="data:image/png;base64,/9j/'
new_src = get_base64_src("wand_guide_step1_1766247369889.png")
if new_src:
    # Use a simpler approach: replace the whole src content
    import re
    html = re.sub(r'<img src="data:image/png;base64,[^"]+"', f'<img src="{new_src}"', html)

# Add Step 4 Visualization diagram
viz_src = get_base64_src("openlpt_gui_3d_viz_1766020081410.png")
if viz_src:
    insertion_point = 'Only "Valid Frames" (where both points are seen in &ge;2 cameras) will be used for calibration.</p>'
    if insertion_point in html:
        viz_html = f'\n        <img src="{viz_src}" alt="3D Visualization">\n        <div class="screenshot-caption">Figure 2: 3D Visualization of camera poses and detected points.</div>'
        html = html.replace(insertion_point, insertion_point + viz_html)

# Add Results diagram
res_src = get_base64_src("calibration_3d_diagram_1766239530493.png")
if res_src:
    insertion_point = 'Target &lt; 0.05mm).</p>'
    if insertion_point in html:
        res_html = f'\n        <img src="{res_src}" alt="Calibration Results 3D Diagram">\n        <div class="screenshot-caption">Figure 3: 3D diagram showing optimized camera alignment and wand paths.</div>'
        html = html.replace(insertion_point, insertion_point + res_html)

# Save
with open(html_path, "w", encoding="utf-8") as f:
    f.write(html)

print("Successfully updated user guide with more visual aids and embedded images.")
