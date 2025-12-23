"""
Image Preprocessing View
Main view for the Image Preprocessing module with Camera Calibration style layout.
"""

import numpy as np
import cv2
import os

from PySide6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QLabel, QFrame,
    QPushButton, QGroupBox, QGridLayout, QSlider, QSpinBox,
    QCheckBox, QComboBox, QFileDialog, QScrollArea, QApplication,
    QLineEdit, QTableWidgetItem
)
from PySide6.QtCore import Qt, Signal
from PySide6.QtGui import QPixmap, QImage, QPainter, QPen, QColor

from .widgets import RangeSlider, ProcessingDialog


def imadjust_opencv(img, low_in, high_in, low_out=0, high_out=255, gamma=1.0):
    """
    img: uint8 or float image
    low_in, high_in, low_out, high_out: same scale as img
    gamma: gamma correction
    """
    # Ensure float for calculation
    img = img.astype(np.float32)

    # normalize to [0,1]
    # Handle division by zero
    diff = high_in - low_in
    if diff < 1e-5:
        diff = 1e-5
        
    img = (img - low_in) / diff
    img = np.clip(img, 0, 1)

    # gamma
    if gamma != 1.0:
        img = img ** gamma

    # scale to output range
    img = img * (high_out - low_out) + low_out
    img = np.clip(img, low_out, high_out)

    return img.astype(np.uint8)


class ZoomableImageLabel(QLabel):
    """
    Label with zoom and pan functionality for image preview.
    Simplified version for preprocessing.
    """
    
    pixelClicked = Signal(int, int, int) # x, y, intensity

    def __init__(self, text="", parent=None):
        super().__init__(text, parent)
        self.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.setMouseTracking(True)
        
        # Image Data
        self._pixmap = None
        self._cv_image = None  # Store original cv2 image for processing
        
        # View State
        self._user_zoom = 1.0
        self._user_pan_x = 0.0
        self._user_pan_y = 0.0
        self.last_mouse_pos = None
        self.is_panning = False
        
    def setPixmap(self, pixmap):
        self._pixmap = pixmap
        self.resetView()
        self.update()
        
    def setCvImage(self, cv_image):
        """Set image from cv2/numpy array."""
        self._cv_image = cv_image
        if cv_image is not None:
            # Convert to QPixmap
            if len(cv_image.shape) == 2:
                # Grayscale
                h, w = cv_image.shape
                bytes_per_line = w
                # Ensure data is contiguous
                if not cv_image.flags['C_CONTIGUOUS']:
                    cv_image = np.ascontiguousarray(cv_image)
                qimg = QImage(cv_image.data, w, h, bytes_per_line, QImage.Format.Format_Grayscale8)
            else:
                # Color (BGR to RGB)
                h, w, ch = cv_image.shape
                rgb = cv2.cvtColor(cv_image, cv2.COLOR_BGR2RGB)
                bytes_per_line = ch * w
                if not rgb.flags['C_CONTIGUOUS']:
                    rgb = np.ascontiguousarray(rgb)
                qimg = QImage(rgb.data, w, h, bytes_per_line, QImage.Format.Format_RGB888)
            self._pixmap = QPixmap.fromImage(qimg)
        else:
            self._pixmap = None
        self.update()
        
    def getCvImage(self):
        """Get the current cv2 image."""
        return self._cv_image
        
    def resetView(self):
        """Reset zoom and pan."""
        self._user_zoom = 1.0
        self._user_pan_x = 0.0
        self._user_pan_y = 0.0
        self.update()
        
    def _calc_transform_params(self):
        """Calculate display transform parameters."""
        if not self._pixmap or self._pixmap.isNull():
            return 1.0, 0, 0
        
        p_w = self._pixmap.width()
        p_h = self._pixmap.height()
        w_w = self.width()
        w_h = self.height()
        
        if p_w <= 0 or p_h <= 0 or w_w <= 0 or w_h <= 0:
            return 1.0, 0, 0
        
        base_scale = min(w_w / p_w, w_h / p_h)
        scale = base_scale * self._user_zoom
        
        t_w = int(p_w * scale)
        t_h = int(p_h * scale)
        
        base_x = (w_w - t_w) / 2
        base_y = (w_h - t_h) / 2
        
        t_x = int(base_x + self._user_pan_x)
        t_y = int(base_y + self._user_pan_y)
        
        return scale, t_x, t_y

    def wheelEvent(self, event):
        """Handle mouse wheel for zooming."""
        if not self._pixmap or self._pixmap.isNull():
            return
            
        mouse_pos = event.position().toPoint()
        delta = event.angleDelta().y()
        zoom_factor = 1.15 if delta > 0 else (1.0 / 1.15)
        
        new_zoom = self._user_zoom * zoom_factor
        new_zoom = max(0.1, min(20.0, new_zoom))
        
        # Zoom towards cursor
        old_scale, old_tx, old_ty = self._calc_transform_params()
        img_x = (mouse_pos.x() - old_tx) / old_scale if old_scale > 0 else 0
        img_y = (mouse_pos.y() - old_ty) / old_scale if old_scale > 0 else 0
        
        self._user_zoom = new_zoom
        
        new_scale, new_tx, new_ty = self._calc_transform_params()
        new_widget_x = img_x * new_scale + new_tx
        new_widget_y = img_y * new_scale + new_ty
        
        self._user_pan_x += mouse_pos.x() - new_widget_x
        self._user_pan_y += mouse_pos.y() - new_widget_y
        
        self.update()
        event.accept()

    def mousePressEvent(self, event):
        if event.button() == Qt.MouseButton.RightButton:
            self.is_panning = True
            self.last_mouse_pos = event.position().toPoint()
            self.setCursor(Qt.CursorShape.ClosedHandCursor)
        elif event.button() == Qt.MouseButton.LeftButton:
            # Handle pixel click for intensity inspection
            if self._cv_image is not None:
                scale, tx, ty = self._calc_transform_params()
                if scale > 0:
                    pos = event.position().toPoint()
                    # Map to image coordinates
                    img_x = int((pos.x() - tx) / scale)
                    img_y = int((pos.y() - ty) / scale)
                    
                    h, w = self._cv_image.shape if len(self._cv_image.shape) == 2 else self._cv_image.shape[:2]
                    
                    if 0 <= img_x < w and 0 <= img_y < h:
                        # Get intensity
                        if len(self._cv_image.shape) == 2:
                            val = self._cv_image[img_y, img_x]
                        else:
                            # Convert to simplified intensity (grayscale equivalent) if color
                            val = int(np.mean(self._cv_image[img_y, img_x]))
                            
                        self.pixelClicked.emit(img_x, img_y, int(val))

    def mouseMoveEvent(self, event):
        if self.is_panning:
            current_pos = event.position().toPoint()
            delta = current_pos - self.last_mouse_pos
            self._user_pan_x += delta.x()
            self._user_pan_y += delta.y()
            self.last_mouse_pos = current_pos
            self.update()

    def mouseReleaseEvent(self, event):
        if event.button() == Qt.MouseButton.RightButton:
            self.is_panning = False
            self.setCursor(Qt.CursorShape.ArrowCursor)

    def paintEvent(self, event):
        painter = QPainter(self)
        painter.fillRect(self.rect(), QColor(13, 17, 23))  # Dark background
        
        if self._pixmap and not self._pixmap.isNull():
            scale, t_x, t_y = self._calc_transform_params()
            
            p_w = self._pixmap.width()
            p_h = self._pixmap.height()
            t_w = int(p_w * scale)
            t_h = int(p_h * scale)
            
            from PySide6.QtCore import QRect
            # Use nearest-neighbor interpolation for pixel-sharp display when zoomed
            painter.setRenderHint(QPainter.RenderHint.SmoothPixmapTransform, False)
            painter.drawPixmap(QRect(t_x, t_y, t_w, t_h), self._pixmap)
        else:
            painter.setPen(QColor(100, 100, 100))
            painter.drawText(self.rect(), Qt.AlignmentFlag.AlignCenter, "Load an image to preview")


class ImagePreprocessingView(QWidget):
    """View for image preprocessing functionality with Camera Calibration style layout."""
    
    def __init__(self):
        super().__init__()
        
        # Data
        self.root_path = ""  # Main directory path
        self.camera_folders = []  # List of camera folder paths
        self.camera_images = {}  # {cam_idx: [image_paths]}
        self.camera_backgrounds = {}  # {cam_idx: background_image}
        self.current_cam = 0
        self.current_frame = 0
        self.original_image = None
        self.processed_image = None
        self.current_view_mode = "original"  # original, processed, background
        
        self._setup_ui()
    
    def _setup_ui(self):
        main_layout = QVBoxLayout(self)
        main_layout.setContentsMargins(0, 0, 0, 0)
        main_layout.setSpacing(10)
        
        # === Title ===
        title = QLabel("Image Preprocessing")
        title.setStyleSheet("font-size: 24px; font-weight: bold; color: #00d4ff; margin-bottom: 10px;")
        main_layout.addWidget(title)
        
        # === Main Content (Left: View, Right: Settings) ===
        content_layout = QHBoxLayout()
        content_layout.setSpacing(16)
        
        # === Left: Image Preview ===
        preview_frame = QFrame()
        preview_frame.setStyleSheet("background-color: #000000; border: 1px solid #333;")
        preview_layout = QVBoxLayout(preview_frame)
        preview_layout.setContentsMargins(8, 8, 8, 8)
        
        # Camera tabs (left-aligned) - FIRST
        self.cam_tabs_layout = QHBoxLayout()
        self.cam_tabs_layout.setSpacing(0)
        self.cam_tabs_layout.setAlignment(Qt.AlignmentFlag.AlignLeft)
        self.cam_buttons = []
        # Create initial 4 camera tabs (will be updated when images loaded)
        self._create_camera_tabs(4)
        preview_layout.addLayout(self.cam_tabs_layout)
        
        # Original/Processed/Background toggle - SECOND (below camera tabs)
        toggle_layout = QHBoxLayout()
        self.original_btn = QPushButton("Original")
        self.original_btn.setCheckable(True)
        self.original_btn.setChecked(True)
        self.original_btn.setStyleSheet("""
            QPushButton { background-color: #00d4ff; color: black; border-radius: 4px; padding: 6px 16px; font-weight: bold; }
            QPushButton:checked { background-color: #00d4ff; }
            QPushButton:!checked { background-color: #333; color: #888; }
        """)
        self.original_btn.clicked.connect(lambda: self._toggle_view("original"))
        
        self.processed_btn = QPushButton("Processed")
        self.processed_btn.setCheckable(True)
        self.processed_btn.setStyleSheet("""
            QPushButton { background-color: #333; color: #888; border-radius: 4px; padding: 6px 16px; font-weight: bold; }
            QPushButton:checked { background-color: #00d4ff; color: black; }
            QPushButton:!checked { background-color: #333; color: #888; }
        """)
        self.processed_btn.clicked.connect(lambda: self._toggle_view("processed"))
        
        self.background_btn = QPushButton("Background")
        self.background_btn.setCheckable(True)
        self.background_btn.setStyleSheet("""
            QPushButton { background-color: #333; color: #888; border-radius: 4px; padding: 6px 16px; font-weight: bold; }
            QPushButton:checked { background-color: #00d4ff; color: black; }
            QPushButton:!checked { background-color: #333; color: #888; }
        """)
        self.background_btn.clicked.connect(lambda: self._toggle_view("background"))
        
        toggle_layout.addStretch()
        toggle_layout.addWidget(self.original_btn)
        toggle_layout.addWidget(self.processed_btn)
        toggle_layout.addWidget(self.background_btn)
        toggle_layout.addStretch()
        preview_layout.addLayout(toggle_layout)
        
        # Image display area (Zoomable)
        self.image_label = ZoomableImageLabel("Load an image to preview")
        self.image_label.setMinimumHeight(500)
        self.image_label.pixelClicked.connect(self._on_pixel_clicked)
        preview_layout.addWidget(self.image_label, stretch=1)
        
        content_layout.addWidget(preview_frame, stretch=2)
        
        # === Right: Settings Panel ===
        settings_scroll = QScrollArea()
        settings_scroll.setWidgetResizable(True)
        settings_scroll.setFrameShape(QFrame.Shape.NoFrame)
        settings_scroll.setMinimumWidth(280)
        settings_scroll.setMaximumWidth(400)
        settings_scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        settings_scroll.setVerticalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAsNeeded)
        settings_scroll.setStyleSheet("""
            QScrollArea { background-color: transparent; }
            QScrollBar:vertical {
                background: #1a1a2e;
                width: 10px;
                border-radius: 5px;
            }
            QScrollBar::handle:vertical {
                background: #444;
                border-radius: 5px;
                min-height: 20px;
            }
            QScrollBar::handle:vertical:hover {
                background: #555;
            }
        """)
        
        settings_widget = QWidget()
        settings_layout = QVBoxLayout(settings_widget)
        settings_layout.setSpacing(12)
        settings_layout.setContentsMargins(0, 0, 10, 0)
        
        # Group box style
        group_style = """
            QGroupBox { 
                background-color: #000; 
                border: 1px solid #444; 
                font-weight: bold; 
                color: #00d4ff; 
                border-radius: 6px; 
                margin-top: 15px;
                padding-top: 15px;
            } 
            QGroupBox::title { 
                subcontrol-origin: margin; 
                left: 10px; 
                padding: 0 5px; 
            }
        """
        
        # === Image Source ===
        source_group = QGroupBox("Image Source")
        source_group.setStyleSheet(group_style)
        source_layout = QVBoxLayout(source_group)
        
        # Num Cameras row
        from PySide6.QtWidgets import QFormLayout, QTableWidget, QTableWidgetItem, QHeaderView
        cam_row = QHBoxLayout()
        cam_label = QLabel("Num Cameras:")
        cam_label.setStyleSheet("color: white;")
        cam_row.addWidget(cam_label)
        self.num_cameras_spin = QSpinBox()
        self.num_cameras_spin.setRange(1, 16)
        self.num_cameras_spin.setValue(4)
        self.num_cameras_spin.setStyleSheet("""
            QSpinBox { 
                background-color: #1a1a2e; 
                color: white; 
                border: 1px solid #444; 
                border-radius: 4px; 
                padding: 5px;
            }
        """)
        cam_row.addWidget(self.num_cameras_spin)
        source_layout.addLayout(cam_row)
        
        import qtawesome as qta
        
        # === Project Path (For Export) ===
        project_label = QLabel("Project Path (for Output):")
        project_label.setStyleSheet("color: white;")
        source_layout.addWidget(project_label)
        
        proj_row = QHBoxLayout()
        self.project_path_input = QLineEdit()
        self.project_path_input.setPlaceholderText("Select project output folder...")
        self.project_path_input.setStyleSheet("background-color: #1a1a2e; color: white; border: 1px solid #444; padding: 5px;")
        proj_row.addWidget(self.project_path_input)
        
        proj_browse_btn = QPushButton("")
        proj_browse_btn.setFixedWidth(40)
        proj_browse_btn.setIcon(qta.icon("fa5s.folder-open", color="white"))
        proj_browse_btn.setStyleSheet("background-color: #333; color: white; border: 1px solid #444;")
        proj_browse_btn.clicked.connect(self._browse_project_path)
        proj_row.addWidget(proj_browse_btn)
        source_layout.addLayout(proj_row)
        
        # Load Images Button
        browse_btn = QPushButton(" Load Images from Folder")
        browse_btn.setIcon(qta.icon("fa5s.images", color="black"))
        browse_btn.setStyleSheet("""
            QPushButton {
                background-color: #00d4ff; 
                color: black; 
                border: 1px solid #00a0cc; 
                border-radius: 4px; 
                padding: 8px;
                font-weight: bold;
            }
            QPushButton:hover { background-color: #66e5ff; }
        """)
        browse_btn.clicked.connect(self._browse_images)
        source_layout.addWidget(browse_btn)
        
        self.image_count_label = QLabel("0 images loaded")
        self.image_count_label.setStyleSheet("color: #a0a0a0;")
        source_layout.addWidget(self.image_count_label)
        
        # Invert checkbox (applied to all image operations)
        self.invert_check = QCheckBox("Invert")
        self.invert_check.setStyleSheet("color: white;")
        source_layout.addWidget(self.invert_check)
        
        # Frame List
        frame_list_label = QLabel("Frame List (Click to Preview):")
        frame_list_label.setStyleSheet("color: white;")
        source_layout.addWidget(frame_list_label)
        
        self.frame_table = QTableWidget()
        self.frame_table.setColumnCount(2)
        self.frame_table.setHorizontalHeaderLabels(["Index", "Filename"])
        self.frame_table.horizontalHeader().setSectionResizeMode(1, QHeaderView.ResizeMode.Stretch)
        self.frame_table.verticalHeader().setVisible(False)
        self.frame_table.setSelectionBehavior(QTableWidget.SelectionBehavior.SelectRows)
        self.frame_table.setSelectionMode(QTableWidget.SelectionMode.SingleSelection)
        self.frame_table.setStyleSheet("background-color: #0d1117; border: 1px solid #333; color: white;")
        self.frame_table.setFixedHeight(120)
        self.frame_table.currentCellChanged.connect(lambda r, c, pr, pc: self._on_frame_clicked(r, c))
        source_layout.addWidget(self.frame_table)
        
        settings_layout.addWidget(source_group)
        
        # === Background Subtraction ===
        bg_group = QGroupBox("Background Subtraction")
        bg_group.setStyleSheet(group_style)
        bg_layout = QGridLayout(bg_group)
        bg_layout.setVerticalSpacing(10)
        
        self.bg_enabled = QCheckBox("Enable")
        self.bg_enabled.setStyleSheet("color: white;")
        bg_layout.addWidget(self.bg_enabled, 0, 0)
        
        # Calculate Background button
        self.calc_bg_btn = QPushButton("Calculate")
        self.calc_bg_btn.setStyleSheet("""
            QPushButton {
                background-color: #00d4ff; 
                color: black; 
                border-radius: 4px; 
                padding: 5px 12px;
                font-weight: bold;
            }
            QPushButton:hover { background-color: #66e5ff; }
        """)
        self.calc_bg_btn.clicked.connect(self._calculate_all_backgrounds)
        bg_layout.addWidget(self.calc_bg_btn, 0, 1)
        
        # Skip Frames
        skip_label = QLabel("Skip Frames:")
        skip_label.setStyleSheet("color: white;")
        bg_layout.addWidget(skip_label, 1, 0)
        self.skip_frames_spin = QSpinBox()
        self.skip_frames_spin.setRange(0, 100)
        self.skip_frames_spin.setValue(5)
        self.skip_frames_spin.setStyleSheet("""
            QSpinBox { 
                background-color: #1a1a2e; 
                color: white; 
                border: 1px solid #444; 
                border-radius: 4px; 
                padding: 5px;
            }
        """)
        bg_layout.addWidget(self.skip_frames_spin, 1, 1)
        
        # Avg Count (short for "total images for average")
        avg_label = QLabel("Avg Count:")
        avg_label.setStyleSheet("color: white;")
        bg_layout.addWidget(avg_label, 2, 0)
        self.avg_count_spin = QSpinBox()
        self.avg_count_spin.setRange(1, 999999)
        self.avg_count_spin.setValue(1000)
        self.avg_count_spin.setStyleSheet("""
            QSpinBox { 
                background-color: #1a1a2e; 
                color: white; 
                border: 1px solid #444; 
                border-radius: 4px; 
                padding: 5px;
            }
        """)
        bg_layout.addWidget(self.avg_count_spin, 2, 1)
        
        settings_layout.addWidget(bg_group)
        
        # === Image Source Info & Pixel Inspector ===
        # Place pixel info here
        self.pixel_info_label = QLabel("Click image to inspect pixel")
        self.pixel_info_label.setStyleSheet("color: #00d4ff; font-size: 11px;")
        self.pixel_info_label.setAlignment(Qt.AlignmentFlag.AlignRight)
        
        # Add to top layout or somewhere visible. 
        # Putting it in main layout top bar or overlay might be complex.
        # Let's put it at the bottom of the settings scroll for now, or inside a group.
        # Actually, let's put it in the "Image Source" group for high visibility
        pass # Just initialization logic, placement happens in layout construction
        
        # ... Re-arrange layouts slightly to fit it ...
        # Let's add it to main settings layout for now
        settings_layout.addWidget(self.pixel_info_label)
        
        # === Image Adjustment ===
        adjust_group = QGroupBox("Intensity Adjustment")
        adjust_group.setStyleSheet(group_style)
        adjust_layout = QGridLayout(adjust_group)
        adjust_layout.setVerticalSpacing(12)  # 10% more spacing
        adjust_layout.setHorizontalSpacing(10)
        
        # Intensity Range Slider (Dual Handle + SpinBoxes)
        range_label = QLabel("Input Range:")
        range_label.setStyleSheet("color: white;")
        adjust_layout.addWidget(range_label, 0, 0, 1, 3)
        
        self.range_slider = RangeSlider(initial_min=0, initial_max=255)
        self.range_slider.rangeChanged.connect(self._on_settings_changed)
        adjust_layout.addWidget(self.range_slider, 1, 0, 1, 3)
        
        # Denoise (LaVision Processing)
        self.denoise_check = QCheckBox("Enhanced Denoise")
        self.denoise_check.setStyleSheet("color: white; font-weight: bold;")
        self.denoise_check.stateChanged.connect(self._on_settings_changed)
        adjust_layout.addWidget(self.denoise_check, 2, 0, 1, 3)
        
        settings_layout.addWidget(adjust_group)
        
        # === Buttons ===
        settings_layout.addStretch()
        
        preview_btn = QPushButton("Preview")
        preview_btn.setStyleSheet("""
            QPushButton {
                background-color: #00d4ff; 
                color: black; 
                border-radius: 4px; 
                padding: 10px;
                font-weight: bold;
                font-size: 14px;
            }
            QPushButton:hover { background-color: #66e5ff; }
        """)
        preview_btn.clicked.connect(self._preview_processing)
        settings_layout.addWidget(preview_btn)
        
        settings_layout.addWidget(preview_btn)
        
        apply_btn = QPushButton("Process Image (Batch Export)")
        apply_btn.setStyleSheet("""
            QPushButton {
                background-color: #2a3f5f; 
                color: white; 
                border: 1px solid #444; 
                border-radius: 4px; 
                padding: 10px;
                font-size: 14px;
            }
            QPushButton:hover { background-color: #3b5278; }
        """)
        apply_btn.clicked.connect(self._on_process_clicked)
        settings_layout.addWidget(apply_btn)
        
        settings_scroll.setWidget(settings_widget)
        content_layout.addWidget(settings_scroll)
        
        main_layout.addLayout(content_layout)
    
    def _create_camera_tabs(self, num_cams):
        """Create or update camera tab buttons."""
        # Clear existing buttons and stretch
        for btn in self.cam_buttons:
            self.cam_tabs_layout.removeWidget(btn)
            btn.deleteLater()
        self.cam_buttons.clear()
        
        # Remove all items from layout (including stretch)
        while self.cam_tabs_layout.count():
            item = self.cam_tabs_layout.takeAt(0)
            if item.widget():
                item.widget().deleteLater()
        
        # Create new buttons
        for i in range(num_cams):
            btn = QPushButton(f"Cam {i + 1}")
            btn.setCheckable(True)
            btn.setChecked(i == self.current_cam)
            btn.setStyleSheet("""
                QPushButton { 
                    background-color: #333; 
                    color: #888; 
                    border: 1px solid #444; 
                    border-radius: 4px; 
                    padding: 6px 16px; 
                    font-weight: bold; 
                    margin-right: 2px;
                }
                QPushButton:checked { 
                    background-color: #444; 
                    color: white; 
                    border-bottom: 2px solid #00d4ff;
                }
                QPushButton:hover { background-color: #3a3a3a; }
            """)
            btn.clicked.connect(lambda checked, idx=i: self._on_cam_tab_clicked(idx))
            self.cam_tabs_layout.addWidget(btn)
            self.cam_buttons.append(btn)
        
        # Add stretch at end for left alignment
        self.cam_tabs_layout.addStretch(1)
    
    def _on_cam_tab_clicked(self, cam_idx):
        """Handle click on camera tab."""
        self.current_cam = cam_idx
        # Update button states
        for i, btn in enumerate(self.cam_buttons):
            btn.setChecked(i == cam_idx)
        self._load_current_image()
    
    def _browse_images(self):
        """Open directory dialog to select main image folder."""
        import os
        
        root_dir = QFileDialog.getExistingDirectory(
            self,
            "Select Main Image Directory",
            ""
        )
        if not root_dir:
            return
        
        self.root_path = root_dir
        # Auto-set project path to parent of root_dir if empty
        if not self.project_path_input.text():
            parent_dir = os.path.dirname(root_dir)
            self.project_path_input.setText(parent_dir)
            
        self._scan_images(root_dir)

    def _browse_project_path(self):
        """Open directory dialog to select project output folder."""
        path = QFileDialog.getExistingDirectory(self, "Select Project Output Directory", "")
        if path:
            self.project_path_input.setText(path)

    def _scan_images(self, root_dir):
        """Scan directory for camera folders and images."""
        import os
        num_cams = self.num_cameras_spin.value()
        
        # Get all subdirectories sorted
        try:
            subdirs = sorted([
                d for d in os.listdir(root_dir) 
                if os.path.isdir(os.path.join(root_dir, d))
            ])
        except Exception as e:
            print(f"Error scanning directory: {e}")
            return
        
        if len(subdirs) < num_cams:
            from PySide6.QtWidgets import QMessageBox
            QMessageBox.warning(
                self, "Warning",
                f"Found only {len(subdirs)} folders, but {num_cams} cameras expected."
            )
            num_cams = min(num_cams, len(subdirs))
        
        self.camera_folders = [os.path.join(root_dir, d) for d in subdirs[:num_cams]]
        
        # Load images for each camera
        self.camera_images = {}
        total_images = 0
        
        for i, folder in enumerate(self.camera_folders):
            # Filter for images
            files = sorted([
                f for f in os.listdir(folder)
                if f.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.tif', '.tiff'))
            ])
            
            full_paths = [os.path.join(folder, f) for f in files]
            self.camera_images[i] = full_paths
            
            if i == 0:
                total_images = len(files)
            else:
                total_images = min(total_images, len(files))
        
        # Update UI
        self.current_cam = 0
        self.current_frame = 0
        self.image_count_label.setText(f"{total_images} images per camera")
        
        # Update Camera Tabs
        self._create_camera_tabs(len(self.camera_folders))
        
        # Update Frame List
        self.frame_table.setRowCount(total_images)
        self.image_paths = [] # Keep track of frame 0 paths for table
        
        if 0 in self.camera_images:
            self.image_paths = self.camera_images[0] # Use cam 1 for list
            # Ensure we only iterate up to setRowCount
            count = min(len(self.image_paths), total_images)
            for row in range(count):
                path = self.image_paths[row]
                fname = os.path.basename(path)
                
                item_idx = QTableWidgetItem(str(row + 1))
                item_idx.setData(Qt.ItemDataRole.ForegroundRole, QColor("white"))
                self.frame_table.setItem(row, 0, item_idx)
                
                item_name = QTableWidgetItem(fname)
                item_name.setData(Qt.ItemDataRole.ForegroundRole, QColor("white"))
                self.frame_table.setItem(row, 1, item_name)
        
        # Load first image
        if total_images > 0:
            self._load_current_image()
        

    
    def _populate_frame_table(self):
        """Populate the frame list table with images from first camera."""
        import os
        from PySide6.QtWidgets import QTableWidgetItem
        
        # Use first camera's images as reference
        if 0 not in self.camera_images:
            return
        
        images = self.camera_images[0]
        self.frame_table.setRowCount(len(images))
        for i, path in enumerate(images):
            idx_item = QTableWidgetItem(str(i))
            filename_item = QTableWidgetItem(os.path.basename(path))
            self.frame_table.setItem(i, 0, idx_item)
            self.frame_table.setItem(i, 1, filename_item)
        
        # Select first row
        if images:
            self.frame_table.selectRow(0)
    
    def _on_frame_clicked(self, row, col):
        """Handle click on frame table row."""
        if self.current_cam in self.camera_images:
            images = self.camera_images[self.current_cam]
            if 0 <= row < len(images):
                self.current_frame = row
                self._load_current_image()
    
    def _load_current_image(self):
        """Load the current image for preview."""
        if self.current_cam not in self.camera_images:
            return
        
        images = self.camera_images[self.current_cam]
        if not images or self.current_frame >= len(images):
            return
        
        path = images[self.current_frame]
        raw_image = cv2.imread(path, cv2.IMREAD_UNCHANGED)
        
        if raw_image is not None:
            # Apply invert if checked
            if self.invert_check.isChecked():
                self.original_image = 255 - raw_image
            else:
                self.original_image = raw_image
            
            self.processed_image = None
            
            # If we are in processed mode, re-run the processing pipeline
            if self.current_view_mode == "processed":
                self._preview_processing()
            else:
                # Otherwise just update the view
                self._toggle_view(self.current_view_mode)
    
    def _toggle_view(self, view_mode):
        """Toggle between original, processed, and background view."""
        self.current_view_mode = view_mode
        self._update_toggle_buttons()
        
        if view_mode == "original":
            if self.original_image is not None:
                self.image_label.setCvImage(self.original_image)
        elif view_mode == "processed":
            # Apply background subtraction if enabled
            self._apply_background_subtraction()
            if self.processed_image is not None:
                self.image_label.setCvImage(self.processed_image)
            elif self.original_image is not None:
                self.image_label.setCvImage(self.original_image)
        elif view_mode == "background":
            # Show the calculated background for current camera
            if self.bg_enabled.isChecked() and self.current_cam in self.camera_backgrounds:
                self.image_label.setCvImage(self.camera_backgrounds[self.current_cam])
            else:
                # No background available
                pass
    
    def _apply_background_subtraction(self):
        """Apply background subtraction to current image."""
        if self.original_image is None:
            return
        
        # Convert to grayscale if needed
        if len(self.original_image.shape) == 3:
            img = cv2.cvtColor(self.original_image, cv2.COLOR_BGR2GRAY)
        else:
            img = self.original_image.copy()
        
        # Check if background subtraction is enabled and background exists
        if self.bg_enabled.isChecked() and self.current_cam in self.camera_backgrounds:
            bg = self.camera_backgrounds[self.current_cam]
            # Subtract and clamp to 0
            result = img.astype(np.float32) - bg.astype(np.float32)
            result = np.clip(result, 0, 255).astype(np.uint8)
            self.processed_image = result
        else:
            self.processed_image = img
    
    def _calculate_all_backgrounds(self):
        """Calculate background for all cameras."""
        # Check if images are loaded
        if not self.camera_images:
            return
        
        from PySide6.QtWidgets import QProgressDialog
        from PySide6.QtCore import Qt
        
        skip = self.skip_frames_spin.value()
        avg_count = self.avg_count_spin.value()
        num_cams = len(self.camera_images)
        
        # Create progress dialog
        progress = QProgressDialog("Calculating backgrounds...", None, 0, num_cams, self)
        progress.setWindowTitle("Background Calculation")
        progress.setWindowModality(Qt.WindowModality.WindowModal)
        progress.setMinimumDuration(0)
        progress.setFixedSize(420, 110)
        progress.setStyleSheet("""
            QProgressDialog {
                padding: 15px;
            }
            QProgressBar {
                min-height: 25px;
                max-height: 25px;
                margin: 10px 15px;
            }
        """)
        progress.show()
        
        for cam_idx in self.camera_images:
            progress.setValue(cam_idx)
            progress.setLabelText(f"Calculating background for Camera {cam_idx + 1}...")
            QApplication.processEvents()
            
            images = self.camera_images[cam_idx]
            if not images:
                continue
            
            # Select frames for averaging
            selected_frames = images[skip:skip + avg_count]
            if not selected_frames:
                selected_frames = images[:avg_count]  # Use available if not enough
            
            # Calculate average background
            accumulator = None
            count = 0
            invert = self.invert_check.isChecked()
            for path in selected_frames:
                img = cv2.imread(path, cv2.IMREAD_UNCHANGED)
                if img is None:
                    continue
                # Apply invert if checked
                if invert:
                    img = 255 - img
                if len(img.shape) == 3:
                    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
                if accumulator is None:
                    accumulator = img.astype(np.float64)
                else:
                    accumulator += img.astype(np.float64)
                count += 1
            
            if count > 0:
                self.camera_backgrounds[cam_idx] = (accumulator / count).astype(np.uint8)
        
        progress.setValue(num_cams)
        print(f"Calculated backgrounds for {len(self.camera_backgrounds)} cameras")
    
    def _update_toggle_buttons(self):
        """Update toggle button states."""
        mode = getattr(self, 'current_view_mode', 'original')
        self.original_btn.setChecked(mode == "original")
        self.processed_btn.setChecked(mode == "processed")
        self.background_btn.setChecked(mode == "background")
    
    def _on_settings_changed(self):
        """Called when any adjustment setting changes."""
        pass  # Preview is manual via button
    
    
    def _apply_processing_pipeline(self, img_data, bg_data=None):
        """
        Apply the full processing pipeline to a single image.
        Pipeline: Invert -> Background Subtraction -> Denoise (LaVision) -> Intensity Adjustment
        """
        # 0. Ensure format
        if len(img_data.shape) == 3:
            gray = cv2.cvtColor(img_data, cv2.COLOR_BGR2GRAY)
        else:
            gray = img_data.copy()
            
        result = gray.astype(np.float32)
        
        # 1. Background Subtraction
        if self.bg_enabled.isChecked() and bg_data is not None:
            result = result - bg_data.astype(np.float32)
            result = np.clip(result, 0, 255)
            
        # 2. LaVision Processing (Enhanced Denoise)
        if self.denoise_check.isChecked():
            a = result.astype(np.float32)
            # Sliding minimum subtraction
            kernel = np.ones((3, 3), np.uint8)
            b = cv2.erode(a, kernel, iterations=1)
            c = a - b
            # Second pass
            b = cv2.erode(a, kernel, iterations=1) # Note: Logic copied from preview, check if intentional
            c = c - b # Effectively c - 2*b if b same? Wait, original logic:
            # c = a - b
            # then c = c - b (where b is recopied). 
            # In MATLAB code provided: c = c - imerode(a, true(3)). So yes, subtract min filter twice.
            
            d = cv2.GaussianBlur(c, (0, 0), 0.5)
            
            k_size = 100
            e = cv2.blur(d, (k_size, k_size))
            f = a - e
            
            # Sharpen
            blurred_f = cv2.GaussianBlur(f, (0, 0), 1.0)
            sharp = f + 0.8 * (f - blurred_f)
            
            result = np.clip(sharp, 0, 255).astype(np.uint8)
            
        # 3. Intensity Adjustment (imadjust)
        low_in = self.range_slider.minValue()
        high_in = self.range_slider.maxValue()
        result = imadjust_opencv(result, low_in, high_in)
        
        return result.astype(np.uint8)
    
    def _preview_processing(self):
        """Apply current settings and show preview."""
        if self.original_image is None:
            return
        
        # Prepare background
        bg = None
        if self.current_cam in self.camera_backgrounds:
            bg = self.camera_backgrounds[self.current_cam]
            
        # Use valid pipeline
        processed = self._apply_processing_pipeline(self.original_image, bg)
        
        self.processed_image = processed
        self.show_processed = True
        self._update_toggle_buttons()
        self.image_label.setCvImage(self.processed_image)
    
    def _apply_to_all(self):
        """Apply current settings to all loaded images."""
        # TODO: Implement batch processing
        print(f"Apply to all {len(self.image_paths)} images")

    def _on_process_clicked(self):
        """Start batch processing of all images."""
        import os
        
        # Validate Project Path
        project_path = self.project_path_input.text().strip()
        if not project_path:
            # Default to parent of loaded images if empty
            if self.root_path:
                project_path = os.path.dirname(self.root_path)
            else:
                return # Should warn user
        
        if not os.path.exists(project_path):
            try:
                os.makedirs(project_path)
            except Exception as e:
                print(f"Error creating project directory: {e}")
                return

        # Prepare structure
        img_file_dir = os.path.join(project_path, "imgFile")
        if not os.path.exists(img_file_dir):
            os.makedirs(img_file_dir)
            
        # Build task list
        # List of (cam_idx, image_path, output_path)
        tasks = []
        
        # Also need to store paths for text file generation
        # cam_files_map[cam_idx] = [abs_path1, abs_path2, ...]
        cam_files_map = {} 
        
        for cam_idx, file_list in self.camera_images.items():
            cam_dir_name = f"cam{cam_idx + 1}"
            cam_out_dir = os.path.join(img_file_dir, cam_dir_name)
            if not os.path.exists(cam_out_dir):
                os.makedirs(cam_out_dir)
            
            cam_files_map[cam_idx] = []
            
            for i, src_path in enumerate(file_list):
                # Naming: img000000.tif
                filename = f"img{i:06d}.tif"
                dst_path = os.path.join(cam_out_dir, filename)
                
                # Add to task list
                tasks.append({
                    "src": src_path,
                    "dst": dst_path,
                    "cam_idx": cam_idx
                })
                
                # Record absolute path for text file
                cam_files_map[cam_idx].append(os.path.abspath(dst_path))

        if not tasks:
            return

        # Start Processing Dialog
        self.processing_dialog = ProcessingDialog(self, title="Batch Processing Images")
        self.processing_dialog.stop_signal.connect(self._stop_batch_processing)
        self.processing_dialog.pause_signal.connect(self._pause_batch_processing)
        self.processing_dialog.show()
        
        # Run processing in a separate thread/loop
        # For simplicity in this PySide implementation without separate worker class logic right here,
        # we will use QApplication.processEvents within a loop, 
        # but optimally should be QThread. 
        # Given the "agentic" constraint, let's implement a robust loop with processEvents
        # or a minimal QThread if needed. 
        # Let's use a simple generator-based approach or instant loop if fast enough, 
        # but image processing is slow.
        # Implementation: Loop with processEvents to keep UI responsive.
        
        self._is_processing = True
        self._is_paused = False
        self._stop_requested = False
        
        total = len(tasks)
        processed_count = 0
        
        for task in tasks:
            if self._stop_requested:
                break
                
            while self._is_paused:
                QApplication.processEvents()
                if self._stop_requested:
                    break
            
            # 1. Load
            src_img = cv2.imread(task["src"], cv2.IMREAD_UNCHANGED)
            if src_img is None:
                continue
                
            # Invert if needed (global setting)
            if self.invert_check.isChecked():
                src_img = 255 - src_img
            
            # 2. Get Background
            bg = self.camera_backgrounds.get(task["cam_idx"])
            
            # 3. Process
            processed = self._apply_processing_pipeline(src_img, bg)
            
            # 4. Save
            cv2.imwrite(task["dst"], processed)
            
            processed_count += 1
            self.processing_dialog.update_progress(processed_count, total)
            QApplication.processEvents()
        
        # Generate Text Files
        if not self._stop_requested:
            for cam_idx, paths in cam_files_map.items():
                # cam0ImageNames.txt (cam indices often 0-based in some configs, user said cam1, cam2 folders...)
                # User request: "cam1, cam2, cam3... folders" AND "cam0ImageNames.txt, cam1ImageNames.txt"
                # This explicitly implies Folder Index = 1-based, Text File Index = 0-based.
                
                txt_filename = f"cam{cam_idx}ImageNames.txt"
                txt_path = os.path.join(img_file_dir, txt_filename)
                with open(txt_path, "w") as f:
                    for p in paths:
                        f.write(p + "\n")
        
        self.processing_dialog.close()
        self._is_processing = False
        
    def _stop_batch_processing(self):
        self._stop_requested = True
        
    def _pause_batch_processing(self, paused):
        self._is_paused = paused

    def _on_pixel_clicked(self, x, y, intensity):
        """Handle pixel click signal."""
        self.pixel_info_label.setText(f"Pixel ({x}, {y}): {intensity}")
