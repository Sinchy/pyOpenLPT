import matplotlib
matplotlib.use('Agg') # MUST be first
import matplotlib.pyplot as plt
import os
import sys
import qtawesome as qta
from PySide6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QLabel, QFrame,
    QPushButton, QGroupBox, QGridLayout, QProgressBar,
    QTextEdit, QSplitter, QTabWidget, QLineEdit, QMessageBox,
    QFileDialog, QProgressDialog, QSlider, QCheckBox, QSpinBox, QScrollArea,
    QTableWidget, QTableWidgetItem, QHeaderView, QAbstractItemView, QButtonGroup
)
from PySide6.QtCore import Qt, QProcess, QIODevice, Signal, Slot, QObject, QThread, QCoreApplication, QSize, QTimer, QPointF
from PySide6.QtGui import QPixmap, QImage, QPainter, QPen, QColor, QMouseEvent, QWheelEvent

class ZoomableLabel(QLabel):
    """
    A QLabel that supports mouse interaction for Zooming and Panning.
    Emits signals to synchronize multiple views.
    """
    zoomed = Signal(float)          # delta (+1 or -1)
    panned = Signal(float, float)   # dx, dy (normalized -1..1)
    resetView = Signal()            # Double click to reset
    
    def __init__(self, parent=None):
        super().__init__(parent)
        self.last_mouse_pos = None
        self.setMouseTracking(True)
        
    def wheelEvent(self, event: QWheelEvent):
        # Angle delta: positive = zoom in, negative = zoom out
        delta = event.angleDelta().y()
        if delta > 0:
            self.zoomed.emit(1.0)
        elif delta < 0:
            self.zoomed.emit(-1.0)
            
    def mousePressEvent(self, event: QMouseEvent):
        if event.button() == Qt.MouseButton.LeftButton:
            self.last_mouse_pos = event.pos()
            
    def mouseMoveEvent(self, event: QMouseEvent):
        if event.buttons() & Qt.MouseButton.LeftButton and self.last_mouse_pos:
            # Calculate delta relative to label size
            dx = event.pos().x() - self.last_mouse_pos.x()
            dy = event.pos().y() - self.last_mouse_pos.y()
            
            # Normalize delta (inverted Y logic if needed, but here standard drag)
            norm_dx = dx / self.width()
            norm_dy = dy / self.height()
            
            self.panned.emit(norm_dx, norm_dy)
            self.last_mouse_pos = event.pos()
            
    def mouseReleaseEvent(self, event: QMouseEvent):
        if event.button() == Qt.MouseButton.LeftButton:
            self.last_mouse_pos = None

    def mouseDoubleClickEvent(self, event: QMouseEvent):
        self.resetView.emit()
import re
import csv
import numpy as np
from matplotlib.figure import Figure
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.backends.backend_qtagg import NavigationToolbar2QT as NavigationToolbar


class TrackLoaderWorker(QObject):
    """Worker class to load tracks in a background thread."""
    # Signal: (track_lengths, track_3d_coords, obj_type, track_2d_coords)
    finished = Signal(object, object, str, object)
    error = Signal(str)

    def __init__(self, proj_dir):
        super().__init__()
        self.proj_dir = proj_dir

    def run(self):
        try:
            track_dir = os.path.join(self.proj_dir, "Results", "ConvergeTrack")
            if not os.path.exists(track_dir):
                self.finished.emit({}, {}, "Tracer")
                return

            # Determine object type from config.txt
            obj_type = "Tracer"
            config_path = os.path.join(self.proj_dir, "config.txt")
            if os.path.exists(config_path):
                with open(config_path, 'r') as f:
                    content = f.read().lower()
                    if "bubble" in content:
                        obj_type = "Bubble"

            def natsort_key(s):
                return [int(text) if text.isdigit() else text.lower()
                        for text in re.split('([0-9]+)', s)]

            def get_sorted_csvs(prefix):
                files = [f for f in os.listdir(track_dir) if f.startswith(prefix) and f.endswith(".csv")]
                return sorted(files, key=natsort_key)

            patterns = ["LongTrackActive", "LongTrackInactive", "ExitTrack"]
            track_lengths = {}
            track_coords = {}
            track_2d_coords = {}  # {track_id: [(frame_id, {cam_idx: (x2d, y2d, r2d)}), ...]}
            max_id_overall = -1

            for pattern in patterns:
                files = get_sorted_csvs(pattern)
                for filename in files:
                    file_path = os.path.join(track_dir, filename)
                    local_max_id_in_file = -1
                    with open(file_path, 'r', newline='') as f:
                        reader = csv.reader(f)
                        header = next(reader, None)
                        if not header: continue
                        
                        # Use the overall max seen so far to shift IDs for THIS file
                        offset = max_id_overall + 1
                        
                        # Detect if this is a Bubble file (has R3D column)
                        is_bubble_file = (obj_type == "Bubble")
                        
                        for row in reader:
                            if not row: continue
                            try:
                                original_id = int(row[0])
                                frame_id = int(row[1])
                                cumulative_id = original_id + offset
                                
                                x, y, z = float(row[2]), float(row[3]), float(row[4])
                                
                                # For Bubble: row[5] is R3D
                                if is_bubble_file and len(row) > 5:
                                    r3d = float(row[5])
                                    point = (frame_id, x, y, z, r3d)
                                else:
                                    point = (frame_id, x, y, z)
                                
                                track_lengths[cumulative_id] = track_lengths.get(cumulative_id, 0) + 1
                                
                                if cumulative_id not in track_coords:
                                    track_coords[cumulative_id] = []
                                track_coords[cumulative_id].append(point)
                                
                                # Parse 2D coords for each camera
                                # Format depends on object type:
                                # Tracer: [ID, F, X, Y, Z, C0_x, C0_y, C1_x, C1_y, ...] -> Start 5, Stride 2
                                # Bubble: [ID, F, X, Y, Z, R3D, C0_x, C0_y, C0_r, C1_x, ...] -> Start 6, Stride 3
                                
                                cam_2d_data = {}
                                
                                if is_bubble_file:
                                    start_col = 6
                                    stride = 3
                                else:
                                    start_col = 5
                                    stride = 2
                                    
                                cam_idx = 0
                                # Ensure we have enough columns for at least x, y
                                while start_col + 1 < len(row):
                                    try:
                                        cam_x = float(row[start_col])
                                        cam_y = float(row[start_col + 1])
                                        
                                        cam_r2d = 0.0
                                        if is_bubble_file and start_col + 2 < len(row):
                                            cam_r2d = float(row[start_col + 2])
                                        
                                        if cam_x > 0 and cam_y > 0:  # Valid 2D detection
                                            cam_2d_data[cam_idx] = (cam_x, cam_y, cam_r2d)
                                    except (ValueError, IndexError):
                                        pass
                                    
                                    start_col += stride
                                    cam_idx += 1
                                
                                if cumulative_id not in track_2d_coords:
                                    track_2d_coords[cumulative_id] = []
                                track_2d_coords[cumulative_id].append((frame_id, cam_2d_data))
                                
                                if original_id > local_max_id_in_file:
                                    local_max_id_in_file = original_id
                            except (ValueError, IndexError):
                                continue
                    
                    # After processing each file, update the global maximum ID
                    if local_max_id_in_file != -1:
                        max_id_overall += (local_max_id_in_file + 1)

            self.finished.emit(track_lengths, track_coords, obj_type, track_2d_coords)
        except Exception as e:
            self.error.emit(str(e))

class TrackingView(QWidget):
    """View for running and monitoring the tracking process."""
    
    def __init__(self, settings_view=None):
        super().__init__()
        self.settings_view = settings_view
        self.process = QProcess(self)
        self.process.readyReadStandardOutput.connect(self._handle_stdout)
        self.process.readyReadStandardError.connect(self._handle_stderr)
        self.process.finished.connect(self._on_process_finished)
        self.log_file = None
        
        # Cache for statistics
        self.cached_proj_path = None
        self.cached_lengths = None
        self.cached_coords = None
        self.is_loading = False
        self.ui_updated = False
        self.last_plotted_count = -1 # Cache to avoid redundant redraws
        
        # Animation state
        self.anim_timer = QTimer(self)
        self.anim_timer.timeout.connect(self._update_animation_frame)
        self.current_anim_frame = 0
        self.anim_frame_range = (0, 0)
        self.anim_active_data = {} # Filtered tracks for animation
        self.is_animating = False
        
        self.current_fig = None
        self.current_ax = None
        self.current_canvas = None
        
        # Global axis limits (computed once from all track data)
        self.global_xlim = None
        self.global_ylim = None
        self.global_zlim = None
        self.cached_obj_type = "Tracer" # Default, updated from worker
        self.cached_tracer_radius = 2.0 # Default radius if config not found
        
        # 2D View State (Synchronized Zoom/Pan)
        # Zoom level: 1.0 = fit, >1.0 = zoomed in
        self.view_2d_zoom = 1.0
        # Center point: (0.5, 0.5) is image center. Range [0.0, 1.0]
        self.view_2d_center = (0.5, 0.5)
        self.active_2d_map = {} # {frame_id: {cam_idx: [(x, y, r, color), ...]}}
        
        self._setup_ui()
    
    def _setup_ui(self):
        main_layout = QVBoxLayout(self)
        main_layout.setContentsMargins(0, 0, 0, 0)
        main_layout.setSpacing(10)

        # Title
        title = QLabel("Tracking")
        title.setStyleSheet("font-size: 24px; font-weight: bold; color: #00d4ff; margin-bottom: 10px;")
        main_layout.addWidget(title)

        # Content Area (Horizontal)
        content_layout = QHBoxLayout()
        content_layout.setSpacing(16)
        
        # 1. Visualization (LEFT)
        vis_frame = QFrame()
        vis_frame.setObjectName("viewFrame")
        vis_frame.setStyleSheet("background-color: #000000; border: none;")
        vis_layout = QVBoxLayout(vis_frame)
        vis_layout.setContentsMargins(0, 0, 0, 0)

        self.vis_tabs = QTabWidget()
        self.vis_tabs.setStyleSheet("""
            QTabWidget::pane { border: 1px solid #333; }
            QTabBar::tab { background: #222; color: #888; padding: 5px; min-width: 80px; }
            QTabBar::tab:selected { background: #444; color: #fff; }
        """)
        
        # 3D View Container
        self.vis_3d_widget = QWidget()
        self.vis_3d_layout = QVBoxLayout(self.vis_3d_widget)
        self.vis_3d_layout.setContentsMargins(0, 0, 0, 0)
        self.vis_3d_label = QLabel("3D Trajectory Visualization")
        self.vis_3d_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.vis_3d_label.setStyleSheet("color: #4a5a6a; font-size: 18px;")
        self.vis_3d_layout.addWidget(self.vis_3d_label)
        self.vis_tabs.addTab(self.vis_3d_widget, "3D View")

        # 2D View Widget with Camera Selection
        self.vis_2d_widget = QWidget()
        vis_2d_main_layout = QVBoxLayout(self.vis_2d_widget)
        vis_2d_main_layout.setContentsMargins(5, 5, 5, 5)
        vis_2d_main_layout.setSpacing(5)
        
        # Camera selection header (centered checkboxes)
        self.cam_select_frame = QFrame()
        self.cam_select_frame.setStyleSheet("background-color: #000000; border: none;")
        self.cam_select_layout = QHBoxLayout(self.cam_select_frame)
        self.cam_select_layout.setContentsMargins(10, 5, 10, 5)
        self.cam_select_layout.addStretch()
        self.cam_checkboxes = []
        self.cam_select_layout.addStretch()
        vis_2d_main_layout.addWidget(self.cam_select_frame)
        
        # Camera grid (adaptive: 1x2 or 2x2)
        self.cam_grid_widget = QWidget()
        self.cam_grid_layout = QGridLayout(self.cam_grid_widget)
        self.cam_grid_layout.setContentsMargins(0, 0, 0, 0)
        self.cam_grid_layout.setSpacing(2)
        
        # Create 4 camera labels
        self.cam_labels = []
        for i in range(4):
            lbl = ZoomableLabel()
            lbl.setAlignment(Qt.AlignmentFlag.AlignCenter)
            lbl.setStyleSheet("background-color: #000; border: 1px solid #333; color: #555;")
            lbl.setText(f"Camera {i+1}")
            lbl.setMinimumSize(200, 150)
            
            # Connect synchronization signals
            lbl.zoomed.connect(self._on_2d_zoom)
            lbl.panned.connect(self._on_2d_pan)
            lbl.resetView.connect(self._on_2d_reset)
            
            # Don't use setScaledContents - we'll scale pixmap manually to preserve aspect ratio
            self.cam_labels.append(lbl)
        
        # Initially hide all labels, will be shown when cameras are selected
        for lbl in self.cam_labels:
            lbl.hide()
        
        vis_2d_main_layout.addWidget(self.cam_grid_widget, stretch=1)
        
        # Placeholder for when no cameras selected
        self.no_cam_label = QLabel("Select cameras to display")
        self.no_cam_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.no_cam_label.setStyleSheet("color: #4a5a6a; font-size: 16px;")
        vis_2d_main_layout.addWidget(self.no_cam_label)
        
        self.vis_tabs.addTab(self.vis_2d_widget, "2D View")
        
        # Track selected cameras and image paths
        self.selected_cams = []
        self.cam_image_paths = {}  # {cam_id: [frame0_path, frame1_path, ...]}
        self.num_cameras = 0

        # Execution Log moved to the left
        self.log_text = QTextEdit()
        self.log_text.setReadOnly(True)
        self.log_text.setStyleSheet("""
            background-color: #0a0a0a; 
            color: #00ff00; 
            font-family: 'Consolas', 'Monaco', monospace;
            font-size: 13px;
            border: 0;
        """)
        self.vis_tabs.addTab(self.log_text, "Execution Log")
        
        vis_layout.addWidget(self.vis_tabs)
        
        # Timeline Slider (Full width)
        self.view_slider = QSlider(Qt.Orientation.Horizontal)
        self.view_slider.setEnabled(False)
        self.view_slider.setStyleSheet("""
            QSlider::groove:horizontal {
                border: 1px solid #333;
                height: 8px;
                background: #111;
                margin: 2px 0;
                border-radius: 4px;
            }
            QSlider::handle:horizontal {
                background: #00d4ff;
                width: 16px;
                margin: -4px 0;
                border-radius: 8px;
            }
            QSlider::handle:horizontal:disabled {
                background: #555;
            }
        """)
        self.view_slider.sliderMoved.connect(self._on_timeline_scrub)
        self.view_slider.valueChanged.connect(self._on_timeline_scrub)
        vis_layout.addWidget(self.view_slider)
        
        content_layout.addWidget(vis_frame, stretch=2)
        
        # 2. Controls (RIGHT)
        self.ctrl_tabs = QTabWidget()
        self.ctrl_tabs.setFixedWidth(370)
        self.ctrl_tabs.setStyleSheet("""
             QTabWidget::pane { border: 1px solid #444; background: #000000; }
             QTabBar::tab { background: #222; color: #aaa; padding: 8px; min-width: 120px; }
             QTabBar::tab:selected { background: #000000; color: #fff; border-top: 2px solid #00d4ff; border-bottom: 0px; font-weight: bold; }
        """)
        
        # Tab 1: Run Tracking
        self.run_tab = QWidget()
        self._setup_run_tab()
        self.ctrl_tabs.addTab(self.run_tab, "Run Tracking")
        
        # Tab 2: Check Tracking (with scroll area)
        self.check_tab_scroll = QScrollArea()
        self.check_tab_scroll.setWidgetResizable(True)
        self.check_tab_scroll.setStyleSheet("QScrollArea { background: #000; border: none; }")
        
        self.check_tab = QWidget()
        check_layout = QVBoxLayout(self.check_tab)
        check_layout.setSpacing(12)
        
        # 1. Track Statistics Section
        stats_group = QGroupBox("Track Statistics")
        stats_layout = QVBoxLayout(stats_group)
        
        # Header: Labels (Left) and Refresh Button (Right)
        header_layout = QHBoxLayout()
        
        stats_text_layout = QVBoxLayout()
        self.total_tracks_label = QLabel("Total Tracks: --")
        self.avg_len_label = QLabel("Avg Track Length: --")
        stats_text_layout.addWidget(self.total_tracks_label)
        stats_text_layout.addWidget(self.avg_len_label)
        header_layout.addLayout(stats_text_layout)
        
        header_layout.addStretch()
        
        self.refresh_stats_btn = QPushButton()
        self.refresh_stats_btn.setIcon(qta.icon("fa5s.sync-alt", color="#00d4ff"))
        self.refresh_stats_btn.setFixedSize(32, 32)
        self.refresh_stats_btn.setToolTip("Refresh Statistics")
        self.refresh_stats_btn.setCursor(Qt.PointingHandCursor)
        self.refresh_stats_btn.setStyleSheet("""
            QPushButton { 
                background-color: #222; 
                border: 1px solid #444; 
                border-radius: 4px; 
            }
            QPushButton:hover { background-color: #333; border-color: #00d4ff; }
            QPushButton:pressed { background-color: #00d4ff; }
        """)
        self.refresh_stats_btn.clicked.connect(lambda: self._load_track_statistics(force=True))
        header_layout.addWidget(self.refresh_stats_btn, alignment=Qt.AlignmentFlag.AlignTop)
        
        stats_layout.addLayout(header_layout)
        
        # Plot area placeholder
        self.histogram_frame = QFrame()
        self.histogram_frame.setMinimumHeight(210) # Reduced from 280
        self.histogram_frame.setStyleSheet("background-color: transparent; border: none;")
        self.hist_layout = QVBoxLayout(self.histogram_frame)
        self.hist_layout.setContentsMargins(0, 5, 0, 0) # Minimal padding
        self.hist_label = QLabel("Length Histogram Placeholder")
        self.hist_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.hist_label.setStyleSheet("color: #666;")
        self.hist_layout.addWidget(self.hist_label)
        stats_layout.addWidget(self.histogram_frame)
        
        check_layout.addWidget(stats_group)
        
        # 2. Visualization Settings Section
        self.vis_settings_group = QGroupBox("Visualization Settings")
        vis_settings_layout = QVBoxLayout(self.vis_settings_group)
        vis_settings_layout.setSpacing(8)
        
        # Option 1: Slider Selection
        self.track_select_slider_cb = QCheckBox("Display tracks:")
        self.track_select_slider_cb.setChecked(True)
        vis_settings_layout.addWidget(self.track_select_slider_cb)
        
        self.track_slider_widget = QWidget()
        slider_layout = QHBoxLayout(self.track_slider_widget)
        slider_layout.setContentsMargins(20, 0, 0, 0)
        
        self.track_slider = QSlider(Qt.Orientation.Horizontal)
        self.track_slider.setRange(1, 1000)
        self.track_slider.setValue(500)
        self.track_slider.setStyleSheet("""
            QSlider::handle:horizontal {
                background: #00d4ff;
                width: 14px;
                border-radius: 7px;
            }
            QSlider::handle:horizontal:disabled {
                background: #555;
            }
        """)
        
        self.track_slider_label = QLabel("500")
        self.track_slider_label.setFixedWidth(40)
        self.track_slider_label.setStyleSheet("color: #00d4ff; font-weight: bold;")
        
        # Real-time Update Connections
        self.track_slider.sliderMoved.connect(self._on_slider_moved)
        self.track_slider.valueChanged.connect(lambda v: self.track_slider_label.setText(str(v)))
        self.track_slider.valueChanged.connect(self._update_active_selection) # Real-time update
        
        slider_layout.addWidget(self.track_slider)
        slider_layout.addWidget(self.track_slider_label)
        vis_settings_layout.addWidget(self.track_slider_widget)
        
        # Option 2: Table Selection
        self.track_select_table_cb = QCheckBox("Select from table:")
        vis_settings_layout.addWidget(self.track_select_table_cb)
        
        # Container for table to allow indentation
        self.table_container = QWidget()
        table_layout = QVBoxLayout(self.table_container)
        table_layout.setContentsMargins(20, 0, 0, 0)
        table_layout.setSpacing(0)

        self.track_table = QTableWidget()
        self.track_table.setColumnCount(3)
        self.track_table.setHorizontalHeaderLabels(["", "Track ID", "Length"])
        self.track_table.verticalHeader().setVisible(False)
        self.track_table.setSelectionBehavior(QAbstractItemView.SelectionBehavior.SelectRows)
        self.track_table.setSortingEnabled(True)
        self.track_table.horizontalHeader().setSectionResizeMode(0, QHeaderView.ResizeMode.ResizeToContents)
        self.track_table.horizontalHeader().setSectionResizeMode(1, QHeaderView.ResizeMode.Stretch)
        self.track_table.horizontalHeader().setSectionResizeMode(2, QHeaderView.ResizeMode.Stretch)
        self.track_table.setMinimumHeight(120) # Approx 4 rows
        self.track_table.setMaximumHeight(200)
        self.track_table.verticalHeader().setDefaultSectionSize(24) # Smaller rows
        self.track_table.setStyleSheet("""
            QTableWidget { background-color: #111; color: #eee; gridline-color: #333; border: 1px solid #444; font-size: 10px; }
            QTableWidget:disabled { background-color: #222; color: #666; border: 1px solid #333; }
            QHeaderView::section { background-color: #222; color: #aaa; padding: 2px; border: 1px solid #333; font-size: 10px; }
            QTableWidget::item { padding: 0px; }
            QTableWidget::item:selected { background-color: #005a8c; color: #fff; }
        """)
        
        # Trigger real-time update on table check
        self.track_table.itemChanged.connect(lambda item: self._update_active_selection())
        
        table_layout.addWidget(self.track_table)
        vis_settings_layout.addWidget(self.table_container)
        
        # Plot Button
        self.plot_tracks_btn = QPushButton("Plot")
        self.plot_tracks_btn.setStyleSheet("""
            QPushButton { 
                background-color: #006064; color: white; border: 1px solid #00838f; 
                padding: 6px; border-radius: 4px; font-weight: bold;
                margin-top: 5px;
            }
            QPushButton:hover { background-color: #00838f; }
            QPushButton:pressed { background-color: #0097a7; }
            QPushButton:disabled { background-color: #333; color: #666; border: 1px solid #444; }
        """)
        self.plot_tracks_btn.clicked.connect(self._on_plot_clicked)
        vis_settings_layout.addWidget(self.plot_tracks_btn)
        
        check_layout.addWidget(self.vis_settings_group)
        
        # Mutually exclusive logic
        self.selection_mode_group = QButtonGroup(self)
        self.selection_mode_group.addButton(self.track_select_slider_cb, 0)
        self.selection_mode_group.addButton(self.track_select_table_cb, 1)
        self.selection_mode_group.setExclusive(True)
        self.selection_mode_group.buttonToggled.connect(self._on_selection_mode_changed)
        
        # Set initial UI state
        self.track_slider_widget.setEnabled(True)
        self.table_container.setEnabled(False)
        self.track_table.setStyleSheet("QTableWidget { background-color: #0a0a0a; color: #555; gridline-color: #222; border: 1px solid #333; }") # Dimmed
        
        # 3. Animation Settings Section
        self.anim_group = QGroupBox("Animation Settings")
        anim_layout = QGridLayout(self.anim_group)
        
        self.anim_enable_check = QCheckBox("Enable Animation")
        self.anim_enable_check.setStyleSheet("color: #aaa;")
        self.anim_enable_check.stateChanged.connect(self._on_anim_enable_changed)
        
        anim_layout.addWidget(self.anim_enable_check, 0, 0, 1, 2)
        
        anim_layout.addWidget(QLabel("Tail Length (frames):"), 1, 0)
        self.anim_tail_spin = QSpinBox()
        self.anim_tail_spin.setRange(1, 500)
        self.anim_tail_spin.setValue(20)
        self.anim_tail_spin.setStyleSheet("background-color: #222; color: #fff; border: 1px solid #444;")
        anim_layout.addWidget(self.anim_tail_spin, 1, 1)
        
        anim_layout.addWidget(QLabel("Animation FPS:"), 2, 0)
        self.anim_fps_spin = QSpinBox()
        self.anim_fps_spin.setRange(1, 1000)
        self.anim_fps_spin.setValue(60)
        self.anim_fps_spin.setStyleSheet("background-color: #222; color: #fff; border: 1px solid #444;")
        anim_layout.addWidget(self.anim_fps_spin, 2, 1)
        
        self.bubble_scale_label = QLabel("Bubble Scale:")
        anim_layout.addWidget(self.bubble_scale_label, 3, 0)
        self.bubble_scale_spin = QSpinBox()
        self.bubble_scale_spin.setRange(1, 10000)
        self.bubble_scale_spin.setValue(100) # Will be auto-computed on data load
        self.bubble_scale_spin.setStyleSheet("background-color: #222; color: #fff; border: 1px solid #444;")
        anim_layout.addWidget(self.bubble_scale_spin, 3, 1)
        # Hide by default, only show for Bubble data
        self.bubble_scale_label.hide()
        self.bubble_scale_spin.hide()
        
        # Slider moved to main view (self.view_slider)
        
        self.anim_btn = QPushButton(" Start Animation")
        self.anim_btn.setIcon(qta.icon("fa5s.play", color="white"))
        self.anim_btn.setFixedHeight(32)
        self.anim_btn.setStyleSheet("""
            QPushButton { background-color: #2e7d32; color: white; font-weight: bold; border-radius: 4px; border: none; }
            QPushButton:hover { background-color: #388e3c; }
            QPushButton:pressed { background-color: #1b5e20; }
            QPushButton:disabled { background-color: #333; color: #666; }
        """)
        self.anim_btn.setEnabled(False)
        self.anim_btn.clicked.connect(self._toggle_animation)
        anim_layout.addWidget(self.anim_btn, 5, 0, 1, 2)
        
        check_layout.addWidget(self.anim_group)
        
        check_layout.addStretch()
        
        # Connect check_tab to scroll area and add to tabs
        self.check_tab_scroll.setWidget(self.check_tab)
        self.ctrl_tabs.addTab(self.check_tab_scroll, "Check Tracking")
        self.ctrl_tabs.currentChanged.connect(lambda i: self._load_track_statistics() if i == 1 else None)
        
        content_layout.addWidget(self.ctrl_tabs)
        main_layout.addLayout(content_layout)

    def _setup_run_tab(self):
        layout = QVBoxLayout(self.run_tab)
        layout.setSpacing(15)

        # Project Path
        path_group = QGroupBox("Project Environment")
        path_layout = QVBoxLayout(path_group)
        
        path_row = QHBoxLayout()
        self.proj_path_edit = QLineEdit()
        self.proj_path_edit.setReadOnly(True)
        self.proj_path_edit.setPlaceholderText("No project directory set...")
        self.proj_path_edit.setStyleSheet("background-color: #1a1a1a; color: #ccc; border: 1px solid #333;")
        
        self.sync_btn = QPushButton()
        self.sync_btn.setIcon(qta.icon("fa5s.folder-open", color="#00d4ff"))
        self.sync_btn.setToolTip("Browse Project Directory")
        self.sync_btn.setFixedSize(30, 30)
        self.sync_btn.clicked.connect(self._browse_project)
        
        path_row.addWidget(self.proj_path_edit)
        path_row.addWidget(self.sync_btn)
        path_layout.addLayout(path_row)
        layout.addWidget(path_group)

        # Execution Controls
        exec_group = QGroupBox("Execution")
        exec_layout = QVBoxLayout(exec_group)
        
        self.run_btn = QPushButton(" Run OpenLPT")
        self.run_btn.setIcon(qta.icon("fa5s.play", color="white"))
        self.run_btn.setFixedHeight(40)
        self.run_btn.setStyleSheet("""
            QPushButton { background-color: #0066cc; color: white; font-weight: bold; border-radius: 4px; }
            QPushButton:hover { background-color: #0077ee; }
            QPushButton:disabled { background-color: #333; color: #666; }
        """)
        self.run_btn.clicked.connect(self._run_tracking)
        
        self.stop_btn = QPushButton(" Stop Execution")
        self.stop_btn.setIcon(qta.icon("fa5s.stop", color="white"))
        self.stop_btn.setFixedHeight(40)
        self.stop_btn.setStyleSheet("""
            QPushButton { background-color: #990000; color: white; font-weight: bold; border-radius: 4px; }
            QPushButton:hover { background-color: #bb0000; }
            QPushButton:disabled { background-color: #333; color: #666; }
        """)
        self.stop_btn.setEnabled(False)
        self.stop_btn.clicked.connect(self._stop_tracking)
        
        exec_layout.addWidget(self.run_btn)
        exec_layout.addWidget(self.stop_btn)
        layout.addWidget(exec_group)

        layout.addStretch()

    def showEvent(self, event):
        super().showEvent(event)
        self._sync_project_path()

    def _sync_project_path(self):
        """Fetch project path from settings view."""
        if self.settings_view and hasattr(self.settings_view, 'project_path'):
            path = self.settings_view.project_path.text()
            if path:
                self.proj_path_edit.setText(path)

    def _browse_project(self):
        """Manually select project directory."""
        current_dir = self.proj_path_edit.text() or os.path.expanduser("~")
        dir_path = QFileDialog.getExistingDirectory(self, "Select Project Directory", current_dir)
        if dir_path:
            self.proj_path_edit.setText(dir_path)

    def _run_tracking(self):
        """Execute OpenLPT.exe with pre-run checks."""
        proj_dir = self.proj_path_edit.text()
        if not proj_dir or not os.path.exists(proj_dir):
            self._append_log("[Error] Project directory not found. Please set it in Settings.\n")
            return

        # 1. Routine Checks
        passed, error_msg = self._check_project_files(proj_dir)
        if not passed:
            self.log_text.clear()
            self.vis_tabs.setCurrentWidget(self.log_text)
            self._append_log(f"[Error] Pre-run check failed:\n{error_msg}\n")
            
            QMessageBox.warning(
                self, 
                "Missing Environment Files",
                f"Tracking cannot start because some required files are missing:\n\n{error_msg}\n\n"
                "Please go back to the 'Settings' module to save the configuration and ensure camera parameters are synchronized."
            )
            return

        config_path = os.path.join(proj_dir, "config.txt")
        
        # Find executable relative to GUI script
        base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
        exe_path = os.path.join(base_dir, "build", "Release", "OpenLPT.exe")
        
        if not os.path.exists(exe_path):
            self._append_log(f"[Error] Executable not found at {exe_path}\n")
            return

        # Prepare log file
        log_path = os.path.join(proj_dir, "log.txt")
        try:
            self.log_file = open(log_path, "w")
            self.log_text.clear()
            self.vis_tabs.setCurrentWidget(self.log_text)
            self._append_log(f"[Info] Running: {exe_path} {config_path}\n")
            self._append_log(f"[Info] Logging to: {log_path}\n\n")
        except Exception as e:
            self._append_log(f"[Error] Failed to create log file: {e}\n")
            return

        # Start process
        self.run_btn.setEnabled(False)
        self.stop_btn.setEnabled(True)
        self.process.start(exe_path, [config_path])

    def _check_project_files(self, proj_dir):
        """Verify existence of mandatory files/folders."""
        errors = []
        
        # Check config.txt
        if not os.path.exists(os.path.join(proj_dir, "config.txt")):
            errors.append("- Master config file (config.txt) is missing.")

        # Check camFile folder and contents
        cam_dir = os.path.join(proj_dir, "camFile")
        if not os.path.isdir(cam_dir):
            errors.append("- Camera parameters directory (camFile) is missing.")
        else:
            txt_files = [f for f in os.listdir(cam_dir) if f.startswith("cam") and f.endswith(".txt")]
            if not txt_files:
                errors.append("- No camera parameter files (cam*.txt) found in camFile.")

        # Check imgFile folder and subfolders
        img_dir = os.path.join(proj_dir, "imgFile")
        if not os.path.isdir(img_dir):
            errors.append("- Preprocessed images directory (imgFile) is missing.")
        else:
            cam_folders = [f for f in os.listdir(img_dir) if f.startswith("cam") and os.path.isdir(os.path.join(img_dir, f))]
            if not cam_folders:
                errors.append("- No camera image folders (cam0, cam1, etc.) found in imgFile.")

        # Check sub-config based on settings
        if self.settings_view:
            obj_type = self.settings_view.obj_type_combo.currentText()
            sub_config = f"{obj_type.lower()}Config.txt"
            if not os.path.exists(os.path.join(proj_dir, sub_config)):
                errors.append(f"- Sub-configuration file ({sub_config}) is missing. Please save configuration again.")

        if errors:
            return False, "\n".join(errors)
        return True, ""

    def _stop_tracking(self):
        if self.process.state() != QProcess.ProcessState.NotRunning:
            self.process.terminate()
            if not self.process.waitForFinished(2000):
                self.process.kill()

    def _handle_stdout(self):
        data = self.process.readAllStandardOutput().data().decode(errors='replace')
        self._append_log(data)

    def _handle_stderr(self):
        data = self.process.readAllStandardError().data().decode(errors='replace')
        self._append_log(data)

    def _append_log(self, text):
        v_scroll = self.log_text.verticalScrollBar()
        # Check if we are currently at the bottom (with a small 10px margin)
        is_at_bottom = v_scroll.value() >= v_scroll.maximum() - 10
        
        # Save cursor position if we are NOT at the bottom
        # so appending doesn't steal focus/context
        self.log_text.insertPlainText(text)
        
        # Auto-scroll only if we were already trailing the log
        if is_at_bottom:
            v_scroll.setValue(v_scroll.maximum())
            
        # Write to file
        if self.log_file:
            self.log_file.write(text)
            self.log_file.flush()

    def _on_process_finished(self, exit_code, exit_status):
        self.run_btn.setEnabled(True)
        self.stop_btn.setEnabled(False)
        
        if self.log_file:
            self.log_file.close()
            self.log_file = None
            
        status_str = "Successfully" if exit_code == 0 else f"with exit code {exit_code}"
        self.log_text.append(f"\n[Info] Tracking finished {status_str}.")

    def _load_track_statistics(self, force=False):
        """Initialize background loading of tracks with caching."""
        if self.is_loading:
            return

        proj_dir = self.proj_path_edit.text()
        if not proj_dir or not os.path.exists(proj_dir):
            return

        # Use cache if already loaded for this project and not forced
        if not force and self.cached_proj_path == proj_dir and self.cached_lengths:
            self._display_statistics(self.cached_lengths, self.cached_coords)
            return

        # If data is different or forced, we need to clear and reload
        if self.cached_proj_path != proj_dir:
            self.ui_updated = False

        track_dir = os.path.join(proj_dir, "Results", "ConvergeTrack")
        if not os.path.exists(track_dir):
            self.total_tracks_label.setText("Total Tracks: 0")
            self.avg_len_label.setText("Avg Track Length: 0.00")
            self.cached_proj_path = proj_dir
            self.cached_lengths = None
            return

        self.is_loading = True

        # Show Loading Dialog
        self.loading_dialog = QProgressDialog("Reading tracks, please wait...", None, 0, 0, self)
        self.loading_dialog.setWindowTitle("Loading Data")
        self.loading_dialog.setWindowModality(Qt.WindowModal)
        self.loading_dialog.setCancelButton(None)
        self.loading_dialog.setMinimumDuration(0)
        self.loading_dialog.show()

        # Setup Thread and Worker
        self.loader_thread = QThread()
        self.loader_worker = TrackLoaderWorker(proj_dir)
        self.loader_worker.moveToThread(self.loader_thread)

        self.loader_thread.started.connect(self.loader_worker.run)
        self.loader_worker.finished.connect(self._on_statistics_loaded)
        self.loader_worker.error.connect(self._on_loader_error)
        
        # Cleanup
        self.loader_worker.finished.connect(self.loader_thread.quit)
        self.loader_worker.finished.connect(self.loader_worker.deleteLater)
        self.loader_thread.finished.connect(self.loader_thread.deleteLater)
        self.loader_worker.error.connect(self.loader_thread.quit)
        self.loader_worker.error.connect(self.loader_worker.deleteLater)

        self.loader_thread.start()

    @Slot(object, object, str, object)
    def _on_statistics_loaded(self, track_lengths, track_coords, obj_type, track_2d_coords):
        """Update cache and UI with loaded statistics."""
        self.is_loading = False
        
        # Update cache
        self.cached_proj_path = self.proj_path_edit.text()
        self.cached_lengths = track_lengths
        self.cached_coords = track_coords
        self.cached_obj_type = obj_type.strip() if obj_type else "Tracer"
        self.cached_2d_coords = track_2d_coords  # Store 2D coords for 2D view overlays
        self.ui_updated = True
        
        
        # Compute global axis limits from ALL tracks
        all_x, all_y, all_z = [], [], []
        for pts in track_coords.values():
            pts_arr = np.array(pts)
            all_x.extend(pts_arr[:, 1])
            all_y.extend(pts_arr[:, 2])
            all_z.extend(pts_arr[:, 3])
        
        if all_x:
            margin = 0.05 # 5% margin
            x_range = max(all_x) - min(all_x)
            y_range = max(all_y) - min(all_y)
            z_range = max(all_z) - min(all_z)
            self.global_xlim = (min(all_x) - x_range * margin, max(all_x) + x_range * margin)
            self.global_ylim = (min(all_y) - y_range * margin, max(all_y) + y_range * margin)
            self.global_zlim = (min(all_z) - z_range * margin, max(all_z) + z_range * margin)
        
        # Auto-compute bubble scale factor based on max R3D
        if obj_type == "Bubble":
            # Show bubble scale controls
            self.bubble_scale_label.show()
            self.bubble_scale_spin.show()
            all_r3d = []
            for pts in track_coords.values():
                for pt in pts:
                    if len(pt) > 4:
                        all_r3d.append(pt[4])
            if all_r3d:
                max_r3d = max(all_r3d)
                # Target: make largest bubble ~200 pixels, so scale = 200 / (max_r3d^2)
                if max_r3d > 0:
                    auto_scale = int(200 / (max_r3d ** 2))
                    auto_scale = max(10, min(5000, auto_scale)) # clamp to reasonable range
                    self.bubble_scale_spin.setValue(auto_scale)
        else:
            # Hide bubble scale controls for Tracer data
            self.bubble_scale_label.hide()
            self.bubble_scale_spin.hide()
        
        # Re-set this to force redraw for new data
        self.last_plotted_count = -1
        
        # Discover cameras for 2D view
        self._discover_cameras()
        
        self._display_statistics(track_lengths, track_coords, force_redraw=True)
        # Populate table only once when data is loaded
        self._populate_track_table(track_lengths)
        
        # Initial selection update
        self._update_active_selection()

    def _display_statistics(self, track_lengths, track_coords, force_redraw=False):
        """Factorized method to update labels and plots with redundancy check."""
        if not self.isVisible():
            return
            
        proj_path = self.proj_path_edit.text()
        
        # ELIMINATE REDUNDANCY: Only proceed if something actually changed or forced
        if not force_redraw and self.ui_updated and \
           proj_path == self.cached_proj_path:
            return

        # Prepare for work: Show dialog only if one isn't already active
        from PySide6.QtWidgets import QProgressDialog
        if not hasattr(self, 'loading_dialog') or self.loading_dialog is None:
            self.loading_dialog = QProgressDialog("Processing data...", None, 0, 0, self)
            self.loading_dialog.setWindowTitle("Please Wait")
            self.loading_dialog.setWindowModality(Qt.WindowModality.WindowModal)
            self.loading_dialog.show()
            QCoreApplication.processEvents()

        # Update cache markers
        self.cached_proj_path = proj_path
        self.ui_updated = True
        
        # Read tracer config for rendering geometry
        self._read_tracer_radius(proj_path)
        
        lengths = list(track_lengths.values())
        if not lengths:
            # ... cleanup logic ...
            self.total_tracks_label.setText("Total Tracks: 0")
            self.avg_len_label.setText("Avg Track Length: 0.00")
            self.track_slider.setRange(1, 100)
            self.track_slider.setValue(1)
            self.track_slider_label.setText("1")
            for layout in [self.hist_layout, self.vis_3d_layout]:
                for i in reversed(range(layout.count())):
                    item = layout.itemAt(i)
                    if item and item.widget():
                        item.widget().setParent(None)
            self.hist_layout.addWidget(self.hist_label)
            self.vis_3d_layout.addWidget(self.vis_3d_label)
            
            # Clear table
            self.track_table.setRowCount(0)
            
            if self.loading_dialog:
                self.loading_dialog.close()
                self.loading_dialog = None
            return

        total_count = len(lengths)
        self.total_tracks_label.setText(f"Total Tracks: {total_count}")
        self.avg_len_label.setText(f"Avg Track Length: {sum(lengths)/total_count:.2f}")

        # Sync slider range and value
        self.track_slider.setRange(1, total_count)
        # Don't change value if already set? Defaults to 500 or max
        if self.track_slider.value() > total_count:
            self.track_slider.setValue(total_count)
            self.track_slider_label.setText(str(total_count))

        self._update_histogram(lengths)
        
        
        # NOTE: _populate_track_table is moved to _on_statistics_loaded to avoid resetting table on every display update
        # self._populate_track_table(track_lengths)
        
        # Trigger initial plot (will use default slider mode)
        self._on_plot_clicked()

        if self.loading_dialog:
            self.loading_dialog.close()
            self.loading_dialog = None

    def _on_loader_error(self, error_msg):
        """Handle errors during track loading."""
        self.is_loading = False
        if hasattr(self, 'loading_dialog'):
            self.loading_dialog.close()
        QMessageBox.critical(self, "Loading Error", f"Failed to load track data:\n{error_msg}")

    def _discover_cameras(self):
        """Discover available cameras from imgFile directory and create checkboxes."""
        proj_path = self.proj_path_edit.text()
        if not proj_path:
            return
        
        img_file_path = os.path.join(proj_path, "imgFile")
        if not os.path.exists(img_file_path):
            return
        
        # Find cam directories (cam0, cam1, cam2, ... or cam1, cam2, ...)
        cam_dirs = []
        for d in sorted(os.listdir(img_file_path)):
            full_path = os.path.join(img_file_path, d)
            if os.path.isdir(full_path) and d.startswith("cam"):
                cam_dirs.append(d)
        
        self.num_cameras = len(cam_dirs)
        
        # Clear existing checkboxes
        for cb in self.cam_checkboxes:
            cb.deleteLater()
        self.cam_checkboxes.clear()
        
        # Remove all widgets from layout (except stretches)
        while self.cam_select_layout.count() > 2:
            item = self.cam_select_layout.takeAt(1)
            if item.widget():
                item.widget().deleteLater()
        
        # Build image path lists from camXImageNames.txt
        self.cam_image_paths.clear()
        
        for cam_dir in cam_dirs:
            # Try to find corresponding ImageNames.txt
            # e.g. cam1 -> cam1ImageNames.txt
            txt_name = f"{cam_dir}ImageNames.txt"
            txt_path = os.path.join(img_file_path, txt_name)
            
            if os.path.exists(txt_path):
                try:
                    with open(txt_path, 'r') as f:
                        lines = [line.strip() for line in f if line.strip()]
                        # Resolve paths (assuming they are relative to project root or absolute)
                        # The user example showed "./test/...", so os.path.abspath handles it if CWD is correct.
                        # Otherwise might need to join with project root if they are relative to it?
                        # For now, trust os.path.abspath relative to current working dir.
                        self.cam_image_paths[cam_dir] = [os.path.abspath(l) for l in lines]
                except Exception as e:
                    print(f"Error reading {txt_name}: {e}")
                    self.cam_image_paths[cam_dir] = []
            
            # Fallback: Scan directory if list is empty
            if not self.cam_image_paths.get(cam_dir):
                cam_path = os.path.join(img_file_path, cam_dir)
                imgs = sorted([f for f in os.listdir(cam_path) if f.endswith(('.tif', '.tiff', '.png', '.jpg'))])
                self.cam_image_paths[cam_dir] = [os.path.join(cam_path, img) for img in imgs]
        
        # Create checkboxes for cameras
        # Map directory names to logical indices (0-based) for data retrieval
        self.cam_name_to_idx = {}
        
        insert_pos = 1  # After first stretch
        for i, cam_dir in enumerate(cam_dirs):
            self.cam_name_to_idx[cam_dir] = i
            
            cb = QCheckBox(cam_dir.capitalize())
            cb.setStyleSheet("color: #aaa; margin: 0 10px;")
            cb.stateChanged.connect(self._on_cam_checkbox_changed)
            self.cam_checkboxes.append(cb)
            self.cam_select_layout.insertWidget(insert_pos, cb)
            insert_pos += 1
        
        # Auto-select first 4 cameras
        for i, cb in enumerate(self.cam_checkboxes[:4]):
            cb.setChecked(True)

    def _on_cam_checkbox_changed(self, state):
        """Handle camera checkbox state change with max 4 limit."""
        # Count selected cameras
        selected = [cb for cb in self.cam_checkboxes if cb.isChecked()]
        
        if len(selected) > 4:
            # Uncheck the last one that was checked
            sender = self.sender()
            if sender and sender.isChecked():
                sender.blockSignals(True)
                sender.setChecked(False)
                sender.blockSignals(False)
                return
        
        # Update selected cameras list
        self.selected_cams = []
        for cb in self.cam_checkboxes:
            if cb.isChecked():
                # Extract camera name from checkbox text (e.g., "Cam1" -> "cam1")
                cam_name = cb.text().lower()
                self.selected_cams.append(cam_name)
        
        # Update grid layout
        self._update_cam_grid()

    def _update_cam_grid(self):
        """Update camera grid layout based on selected cameras."""
        # Clear grid and remove all stretch factors
        for lbl in self.cam_labels:
            lbl.hide()
            self.cam_grid_layout.removeWidget(lbl)
        
        # Reset all stretch factors
        for row in range(2):
            self.cam_grid_layout.setRowStretch(row, 0)
        for col in range(2):
            self.cam_grid_layout.setColumnStretch(col, 0)
        
        num_selected = len(self.selected_cams)
        
        if num_selected == 0:
            self.no_cam_label.show()
            return
        
        self.no_cam_label.hide()
        
        # Adaptive layout: 1x2 for 1-2 cams, 2x2 for 3-4 cams
        if num_selected <= 2:
            # 1 row, N columns - set equal column stretch
            for i, cam_name in enumerate(self.selected_cams):
                self.cam_labels[i].setText(cam_name.capitalize())
                self.cam_labels[i].show()
                self.cam_grid_layout.addWidget(self.cam_labels[i], 0, i)
                self.cam_grid_layout.setColumnStretch(i, 1)
            self.cam_grid_layout.setRowStretch(0, 1)
        else:
            # 2 rows, 2 columns - set equal stretch for all
            positions = [(0, 0), (0, 1), (1, 0), (1, 1)]
            for i, cam_name in enumerate(self.selected_cams):
                row, col = positions[i]
                self.cam_labels[i].setText(cam_name.capitalize())
                self.cam_labels[i].show()
                self.cam_grid_layout.addWidget(self.cam_labels[i], row, col)
            # Set equal stretch for 2x2 grid
            self.cam_grid_layout.setRowStretch(0, 1)
            self.cam_grid_layout.setRowStretch(1, 1)
            self.cam_grid_layout.setColumnStretch(0, 1)
            self.cam_grid_layout.setColumnStretch(0, 1)
            self.cam_grid_layout.setColumnStretch(1, 1)


    def _update_active_selection(self):
        """Update self.anim_active_data based on current selection signals."""
        if not self.cached_coords or not self.cached_lengths:
            return

        selected_ids = []
        if self.track_select_slider_cb.isChecked():
            # Slider Mode
            count = self.track_slider.value()
            sorted_track_ids = sorted(self.cached_lengths.keys(), key=lambda x: self.cached_lengths[x], reverse=True)
            selected_ids = sorted_track_ids[:count]
        else:
            # Table Mode
            for row in range(self.track_table.rowCount()):
                item = self.track_table.item(row, 0)
                if item.checkState() == Qt.CheckState.Checked:
                    track_id = int(self.track_table.item(row, 1).text())
                    selected_ids.append(track_id)
        
        # Update Animation Data immediately
        self.anim_active_data = {tid: np.array(self.cached_coords[tid]) for tid in selected_ids}
        
        # --- 2D Optimization: Build Frame-Centric Map ---
        # Structure: map[frame][cam_idx] = list of (x, y, r, is_head_candidate)
        # Actually just (x, y, r). We determine 'head' vs 'tail' by frame difference during render.
        self.active_2d_map = {}
        
        if self.anim_active_data and hasattr(self, 'cached_2d_coords'):
            for tid in selected_ids:
                if tid not in self.cached_2d_coords:
                    continue
                # cached_2d_coords[tid] is list of (frame, {cam_idx: (x,y,r)})
                for frame_id, cam_data in self.cached_2d_coords[tid]:
                    if frame_id not in self.active_2d_map:
                        self.active_2d_map[frame_id] = {}
                    
                    for c_idx, coords in cam_data.items():
                        if c_idx not in self.active_2d_map[frame_id]:
                            self.active_2d_map[frame_id][c_idx] = []
                        # coords is (x, y, r)
                        self.active_2d_map[frame_id][c_idx].append(coords)

        # Recalculate animation frame range based on selection
        if self.anim_active_data:
            all_frames = []
            for tid, data in self.anim_active_data.items():
                if data.size > 0:
                    all_frames.append(data[0, 0]) # Start frame
                    all_frames.append(data[-1, 0]) # End frame
            
            if all_frames:
                min_f = int(min(all_frames))
                max_f = int(max(all_frames))
                self.anim_frame_range = (min_f, max_f)
                
                # Reset to start of new range
                self.current_anim_frame = min_f
            else:
                self.anim_frame_range = (0, 100)
        else:
             # Default/Empty state
             self.anim_frame_range = (0, 100)
             self.current_anim_frame = 0
             
        # Ensure 2D view updates if active
        if self.vis_tabs.currentIndex() == 1:
            self._update_2d_view_frame()


    def _on_slider_moved(self, value):
        """Update slider label while dragging."""
        self.track_slider_label.setText(str(value))

    def _on_selection_mode_changed(self, button, checked):
        if not checked: return
        is_table_mode = (button == self.track_select_table_cb)
        self.track_slider_widget.setEnabled(not is_table_mode)
        self.table_container.setEnabled(is_table_mode)
        
        # Visual feedback for table
        style = "background-color: #111; color: #eee; gridline-color: #333; border: 1px solid #444;" if is_table_mode else "background-color: #0a0a0a; color: #555; gridline-color: #222; border: 1px solid #333;"
        self.track_table.setStyleSheet(f"QTableWidget {{ {style} }} QHeaderView::section {{ background-color: #222; color: #aaa; padding: 4px; border: 1px solid #333; }}")

    def _populate_track_table(self, track_lengths):
        """Populate the track selection table with data."""
        self.track_table.blockSignals(True) # Prevent spamming itemChanged
        self.track_table.setRowCount(0)
        self.track_table.setSortingEnabled(False) # Optimize insertion
        
        sorted_ids = sorted(track_lengths.keys(), key=lambda x: track_lengths[x], reverse=True)
        self.track_table.setRowCount(len(sorted_ids))
        
        for row, tid in enumerate(sorted_ids):
            length = track_lengths[tid]
            
            # Checkbox item
            chk_item = QTableWidgetItem()
            chk_item.setFlags(Qt.ItemFlag.ItemIsUserCheckable | Qt.ItemFlag.ItemIsEnabled | Qt.ItemFlag.ItemIsSelectable)
            chk_item.setCheckState(Qt.CheckState.Unchecked)
            self.track_table.setItem(row, 0, chk_item)
            
            # ID item
            id_item = QTableWidgetItem(str(tid))
            id_item.setData(Qt.ItemDataRole.DisplayRole, tid) # For proper sorting
            self.track_table.setItem(row, 1, id_item)
            
            # Length item
            len_item = QTableWidgetItem(str(length))
            len_item.setData(Qt.ItemDataRole.DisplayRole, length) # For proper sorting
            self.track_table.setItem(row, 2, len_item)
            
        self.track_table.setSortingEnabled(True)
        self.track_table.blockSignals(False)

    def _on_plot_clicked(self):
        """Update 3D view based on selected tracks (Static Plot)."""
        if not self.cached_coords: return
        
        # Prepare loading indicator
        from PySide6.QtWidgets import QProgressDialog
        loading = QProgressDialog("Plotting selected data...", None, 0, 0, self)
        loading.setWindowModality(Qt.WindowModality.WindowModal)
        loading.setMinimumDuration(0)
        loading.show()
        QCoreApplication.processEvents()
        
        try:
            # Ensure anim_active_data is up to date (it should be via signals, but good safety)
            if not getattr(self, 'anim_active_data', None):
                self._update_active_selection()
            
            # Use active data for static plot
            if self.anim_active_data:
                plot_coords = {tid: self.cached_coords[tid] for tid in self.anim_active_data.keys()}
                self._update_3d_view(plot_coords)
            
            # Force update 2D if active
            if self.vis_tabs.currentIndex() == 1:
                self._update_2d_view_frame()
                
            # Update cache marker
            self.last_plotted_count = len(self.anim_active_data) if self.anim_active_data else 0
                
        finally:
            loading.close()


    def _toggle_animation(self):
        """Start or stop the 3D trajectory animation."""
        if self.is_animating:
            # STOP
            self.anim_timer.stop()
            self.is_animating = False
            self.anim_btn.setText(" Start Animation")
            self.anim_btn.setIcon(qta.icon("fa5s.play", color="white"))
            self.anim_btn.setStyleSheet("background-color: #2e7d32; color: white; font-weight: bold; border-radius: 4px; border: none;")
            
            # Re-enable settings
            self.vis_settings_group.setEnabled(True)
            
            # Back to static view
            # User request: Do NOT reset to full tracks. Keep the last frame (Pause behavior).
            # if self.anim_active_data:
            #      plot_coords = {tid: self.cached_coords[tid] for tid in self.anim_active_data.keys()}
            #      self._update_3d_view(plot_coords)
            
        else:
            # START
            if not self.cached_coords or not self.anim_active_data:
                QMessageBox.warning(self, "No Data", "Please plot tracks first before tracking.")
                return
            
            # Use existing self.anim_active_data determined by Plot button!
            # Removed re-calculation logic here.
            
            # Determine global frame range
            all_f = []
            for pts in self.anim_active_data.values():
                all_f.extend(pts[:, 0])
            
            if not all_f: return
            
            self.anim_frame_range = (int(min(all_f)), int(max(all_f)))
            self.current_anim_frame = self.anim_frame_range[0]
            
            # Set timer based on FPS
            fps = self.anim_fps_spin.value()
            self.anim_timer.start(int(1000 / fps))
            
            # Init Slider
            self.view_slider.blockSignals(True)
            self.view_slider.setRange(self.anim_frame_range[0], self.anim_frame_range[1])
            self.view_slider.setValue(self.current_anim_frame)
            self.view_slider.setEnabled(True)
            self.view_slider.blockSignals(False)
            
            self.is_animating = True
            self.anim_btn.setText(" Stop Animation")
            self.anim_btn.setIcon(qta.icon("fa5s.stop", color="white"))
            self.anim_btn.setStyleSheet("background-color: #990000; color: white; font-weight: bold; border-radius: 4px; border: none;")
            
            # Disable settings to prevent conflicts
            self.vis_settings_group.setEnabled(False)
            
            # Init persistent scatter references to None if not exists
            if not hasattr(self, 'scatter_tail'): self.scatter_tail = None
            if not hasattr(self, 'scatter_head'): self.scatter_head = None
            # Force reset to ensure clean start
            self.scatter_tail = None
            self.scatter_head = None

    def _read_tracer_radius(self, proj_path):
        """Try to read particle radius used in tracking from tracerConfig.txt."""
        try:
            # Check project root first (standard structure)
            config_path = os.path.join(proj_path, "tracerConfig.txt")
            
            if not os.path.exists(config_path):
                # Fallback: check parent if proj_path was actually a subdir
                parent_dir = os.path.dirname(proj_path)
                config_path = os.path.join(parent_dir, "tracerConfig.txt")
                
            if not os.path.exists(config_path):
                return
                
            with open(config_path, 'r') as f:
                lines = f.readlines()
                # Radius is typically the last line: "2 # Particle radius [px]..."
                # Or we search for the comment
                for line in lines:
                    if "Particle radius" in line:
                        parts = line.split('#')[0].strip()
                        if parts:
                            radius = float(parts)
                            self.cached_tracer_radius = radius
                            break
        except Exception as e:
            print(f"Error reading tracer config: {e}")

    def _update_animation_frame(self):
        """Rendering loop for the animation."""
        if not self.is_animating:
            return
        
        # Check which tab is active
        current_vis_tab = self.vis_tabs.currentIndex()
        
        # Only update 3D if 3D View tab is active (index 0)
        if current_vis_tab == 0 and self.current_ax:
            self._render_3d_frame()
        
        # Update 2D if 2D View tab is active (index 1)
        if current_vis_tab == 1:
            self._update_2d_view_frame()
        
        # Advance frame regardless of which tab is active
        self.current_anim_frame += 1
        if self.current_anim_frame > self.anim_frame_range[1]:
            self.current_anim_frame = self.anim_frame_range[0]
            
        # Update Slider (Block signals to prevent loop)
        self.view_slider.blockSignals(True)
        self.view_slider.setValue(self.current_anim_frame)
        self.view_slider.blockSignals(False)

    def _render_3d_frame(self):
        """Render a single 3D animation frame using persistent scatter artists."""
        ax = self.current_ax
        
        # --- Initialization (First Run) ---
        if not hasattr(self, 'scatter_tail') or self.scatter_tail is None:
            ax.clear()
            ax.set_facecolor('#000000')
            
            # Static style setup
            ax.set_xlabel('X (mm)', color='white', fontsize=10)
            ax.set_ylabel('Y (mm)', color='white', fontsize=10)
            ax.set_zlabel('Z (mm)', color='white', fontsize=10)
            ax.tick_params(colors='white', labelsize=8)
            for pane in [ax.xaxis.pane, ax.yaxis.pane, ax.zaxis.pane]:
                pane.set_facecolor('#000000')
                pane.set_edgecolor('#333')
            ax.grid(True, color='#222')
            
            # Apply fixed axis limits (prevents jittering)
            if self.global_xlim:
                ax.set_xlim3d(self.global_xlim)
                ax.set_ylim3d(self.global_ylim)
                ax.set_zlim3d(self.global_zlim)
                
            # Create persistent scatter objects (empty initially)
            # Tail: Fading blue dots
            self.scatter_tail = ax.scatter([], [], [], c='blue', s=4, depthshade=False)
            
            # Head: Bright white/blue markers
            self.scatter_head = ax.scatter([], [], [], c='white', s=8, alpha=1.0, edgecolors='none')
            
        # --- Update Data ---
        tail_len = self.anim_tail_spin.value()
        curr_f = self.current_anim_frame
        
        all_x, all_y, all_z = [], [], []
        all_alphas = []
        
        lead_x, lead_y, lead_z = [], [], []
        lead_radii = [] 
        
        # Optimize: Iterate active data
        # (This vector parsing is still somewhat costly, but 50x faster than ax.clear)
        for tid, pts in self.anim_active_data.items():
             # pts is [frame, x, y, z] or [frame, x, y, z, r3d]
            mask = (pts[:, 0] <= curr_f) & (pts[:, 0] > curr_f - tail_len)
            tail_pts = pts[mask]
            
            if len(tail_pts) == 0:
                continue
            
            all_x.extend(tail_pts[:, 1])
            all_y.extend(tail_pts[:, 2])
            all_z.extend(tail_pts[:, 3])
            
            # Alphas: Use squared falloff for more obvious gradient
            ages = curr_f - tail_pts[:, 0]
            # Normalize age 0..1
            norm_ages = ages / tail_len
            # Alpha = (1 - age)^2 leads to steeper density dropoff
            track_alphas = np.maximum(0.0, (1.0 - norm_ages) ** 2)
            all_alphas.extend(track_alphas)
            
            if tail_pts[-1, 0] == curr_f:
                lead_x.append(tail_pts[-1, 1])
                lead_y.append(tail_pts[-1, 2])
                lead_z.append(tail_pts[-1, 3])
                if pts.shape[1] >= 5:
                    lead_radii.append(tail_pts[-1, 4])
                else:
                    lead_radii.append(None)

        # Update Tail
        if all_x:
            # Ensure numpy arrays for offsets (critical for 3D)
            self.scatter_tail._offsets3d = (np.array(all_x), np.array(all_y), np.array(all_z))
            # Update sizes to match point count (required for 3D projection sorting)
            self.scatter_tail.set_sizes(np.full(len(all_x), 4))
            # Update colors
            colors = np.zeros((len(all_x), 4))
            colors[:, 0] = 0.0; colors[:, 1] = 0.8; colors[:, 2] = 1.0; colors[:, 3] = all_alphas
            self.scatter_tail.set_array(None) # Disable scaler mapping
            self.scatter_tail.set_facecolors(colors)
            self.scatter_tail.set_edgecolors('none')
        else:
            self.scatter_tail._offsets3d = ([], [], [])
            self.scatter_tail.set_sizes(np.array([]))
            
        # Update Head
        if lead_x:
            self.scatter_head._offsets3d = (np.array(lead_x), np.array(lead_y), np.array(lead_z))
            
            is_bubble = getattr(self, 'cached_obj_type', 'Tracer') == "Bubble"
            if is_bubble:
                 scale = self.bubble_scale_spin.value()
                 sizes = [max(20, ((r if r else 1) ** 2) * scale) for r in lead_radii]
                 self.scatter_head.set_sizes(np.array(sizes))
                 # Brighter cyan, higher opacity
                 self.scatter_head.set_facecolors('#88ffff')
                 self.scatter_head.set_edgecolors('#00aaff')
                 self.scatter_head.set_alpha(0.9)
            else:
                 self.scatter_head.set_sizes(np.full(len(lead_x), 8))
                 self.scatter_head.set_facecolors('white')
                 self.scatter_head.set_edgecolors('none')
                 self.scatter_head.set_alpha(1.0)
        else:
            self.scatter_head._offsets3d = ([], [], [])
            self.scatter_head.set_sizes(np.array([]))
        
        # Light update
        self.current_canvas.draw_idle()

    def _update_2d_view_frame(self):
        """Load and display camera frames for current animation frame."""
        try:
            if not self.selected_cams or not self.cam_image_paths:
                return
            
            # Safety check for animation state
            if not hasattr(self, 'anim_frame_range') or not self.anim_frame_range:
                return
            
            curr_f = self.current_anim_frame
            # No longer shifting by anim_frame_range[0] for images, 
            # assuming list index == frame number as per user requirement.
            
            for i, cam_name in enumerate(self.selected_cams):
                if i >= 4:
                    break
                
                paths = self.cam_image_paths.get(cam_name, [])
                if not paths or curr_f < 0 or curr_f >= len(paths):
                    self.cam_labels[i].setText(f"{cam_name} (Frame {curr_f} N/A)")
                    # Clear previous image to avoid confusion? Or keep last valid?
                    # Keeping last valid might be confusing if frame jump is large.
                    # Let's clear it or show blank.
                    self.cam_labels[i].setPixmap(QPixmap()) 
                    continue
                
                img_path = paths[curr_f]
                
                try:
                    # Use OpenCV for robust image reading (handles 16-bit TIF)
                    import cv2
                    img = cv2.imread(img_path, cv2.IMREAD_UNCHANGED)
                    
                    if img is None:
                        self.cam_labels[i].setText(f"{cam_name} (cv2 read failed)")
                        continue
                    
                    # Normalize 16-bit to 8-bit
                    if img.dtype == np.uint16:
                        img = ((img.astype(np.float32) - img.min()) / (img.max() - img.min() + 1e-6) * 255).astype(np.uint8)
                    
                    # Convert grayscale to RGB
                    if len(img.shape) == 2:
                        img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
                    elif img.shape[2] == 4:
                        img = cv2.cvtColor(img, cv2.COLOR_BGRA2RGB)
                    else:
                        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                    
                    
                    # --- Draw 2D overlays ---
                    # Use logical index from mapping instead of parsing "camX"
                    # This handles "cam1", "cam2"... mapping to data indices 0, 1...
                    try:
                        # Default to i (loop index) if mapping fails, though it shouldn't
                        cam_idx = self.cam_name_to_idx.get(cam_name, i)
                    except:
                        cam_idx = i
                    
                    tail_len = self.anim_tail_spin.value()
                    is_bubble = getattr(self, 'cached_obj_type', 'Tracer') == 'Bubble'
                    
                    # --- Optimized 2D Drawing ---
                    # Instead of burning into pixels, collect points for QPainter overlay
                    points_to_draw = [] # list of (x, y, r, color, is_head)
                    
                    if hasattr(self, 'active_2d_map'):
                        start_f = curr_f - tail_len
                        if start_f < 0: start_f = 0
                        
                        for f in range(start_f, curr_f + 1):
                            if f not in self.active_2d_map: continue
                            
                            cam_points = self.active_2d_map[f].get(cam_idx, [])
                            
                            for (x2d, y2d, r2d) in cam_points:
                                if f == curr_f:
                                    # Head
                                    color = (0, 255, 255) # Cyan
                                    # Use exact float radius for bubbles, default 0 for tracer (will trigger fixed 4.0 later)
                                    if is_bubble and r2d > 0:
                                        radius = float(r2d)
                                    else:
                                        radius = 0 
                                    points_to_draw.append((x2d, y2d, radius, color, True))
                                else:
                                    # Trail
                                    age = curr_f - f
                                    norm_age = age / (tail_len + 1)
                                    alpha = max(0.0, (1.0 - norm_age) ** 2)
                                    g_val = int(255 * alpha)
                                    color = (0, g_val, 0)
                                    points_to_draw.append((x2d, y2d, 0, color, False))
                    
                    
                    if not img.flags['C_CONTIGUOUS']:
                        img = np.ascontiguousarray(img)
                        
                    h_full, w_full, ch = img.shape
                    
                    # Handle "Follow Mode" Overrides or Manual Zoom/Pan
                        
                    # Handle "Follow Mode" Overrides or Manual Zoom/Pan
                    src_x, src_y = 0, 0
                    src_w, src_h = w_full, h_full 
                    
                    # Follow Mode: Single Track Active
                    is_follow_mode = False
                    if len(self.anim_active_data) == 1 and hasattr(self, 'active_2d_map') and curr_f in self.active_2d_map:
                         # Get the single track's projected 2D position for this camera
                         # active_2d_map structure: {frame: {cam_idx: [(x, y, r), ...]}}
                         cam_points = self.active_2d_map[curr_f].get(cam_idx, [])
                         if cam_points:
                             # Assume first point is the target (since only 1 track)
                             target_x, target_y, target_r = cam_points[0]
                             is_follow_mode = True
                             
                             # Determine Window Size
                             # Bubble: target_r > 0 -> Window = 20 * r
                             # Tracer: target_r == 0 -> Window = 20 * 4 = 80px (based on default 4px radius)
                             # User requested "10x diameter" which is 20x radius.
                             
                             if target_r > 0:
                                 half_w = 10 * target_r
                             else:
                                 # Fallback to config or default 4.0
                                 base_r = self.cached_tracer_radius if hasattr(self, 'cached_tracer_radius') else 4.0
                                 half_w = 10 * base_r
                             
                             # Ensure minimum size to prevent 0-size crash
                             half_w = max(10, int(half_w))
                             half_h = half_w # Square window
                             
                             # Calculate Crop Box
                             cx, cy = int(target_x), int(target_y)
                             x1 = cx - half_w
                             y1 = cy - half_h
                             x2 = cx + half_w
                             y2 = cy + half_h
                             
                             # Pad with black if out of bounds (User Request: Keep object centered)
                             # We need to extract the ROI from valid image area and place it into a black canvas.
                             
                             crop_w = x2 - x1
                             crop_h = y2 - y1
                             final_img = np.zeros((crop_h, crop_w, ch), dtype=img.dtype)
                             
                             # Calculate intersection with image
                             ix1 = max(0, x1); iy1 = max(0, y1)
                             ix2 = min(w_full, x2); iy2 = min(h_full, y2)
                             
                             if ix2 > ix1 and iy2 > iy1:
                                 # Copy valid region to canvas
                                 # Compute offsets in canvas
                                 ox = ix1 - x1
                                 oy = iy1 - y1
                                 
                                 valid_patch = img[iy1:iy2, ix1:ix2]
                                 final_img[oy:oy+(iy2-iy1), ox:ox+(ix2-ix1)] = valid_patch
                             
                             img = final_img
                             
                             # Override source rects for marker drawing later
                             # src_x/y is top-left of the view in image coordinates
                             src_x = x1
                             src_y = y1
                             src_w = crop_w
                             src_h = crop_h
                    
                    if not is_follow_mode and self.view_2d_zoom > 1.0:
                         uv_w = 1.0 / self.view_2d_zoom
                         uv_h = 1.0 / self.view_2d_zoom
                         cx, cy = self.view_2d_center
                         src_x = int((cx - uv_w / 2) * w_full)
                         src_y = int((cy - uv_h / 2) * h_full)
                         src_x = max(0, src_x)
                         src_y = max(0, src_y)
                         # Clamp crop directly for manual zoom (standard behavior)
                         x2 = min(w_full, src_x + int(uv_w * w_full))
                         y2 = min(h_full, src_y + int(uv_h * h_full))
                         # Recalc w/h
                         src_w = x2 - src_x
                         src_h = y2 - src_y
                         
                         img = img[src_y:y2, src_x:x2]

                    # QImage requires C-contiguous
                    if not img.flags['C_CONTIGUOUS']:
                        img = np.ascontiguousarray(img)

                    h, w, ch = img.shape
                    bytes_per_line = ch * w
                    qimg = QImage(img.data, w, h, bytes_per_line, QImage.Format.Format_RGB888).copy()
                    pixmap = QPixmap.fromImage(qimg)
                    
                    if not pixmap.isNull():
                        # Scale to fit label while preserving aspect ratio
                        label_size = self.cam_labels[i].size()
                        # User Request: Show raw pixels (Nearest Neighbor), do not smooth.
                        scaled = pixmap.scaled(label_size, Qt.AspectRatioMode.KeepAspectRatio, Qt.TransformationMode.FastTransformation)
                        
                        # --- QPainter Overlay for Constant Sized Markers ---
                        painter = QPainter(scaled)
                        painter.setRenderHint(QPainter.RenderHint.Antialiasing)
                        
                        # Default to full image for "No Zoom" case, unless overridden
                        if not is_follow_mode:
                            src_x, src_y = 0, 0
                            src_w, src_h = w_full, h_full 
                         
                        # Re-calculate correct source rect for projection
                        if is_follow_mode:
                            # We already computed these above
                            pass # variables src_x, src_y, src_w, src_h are set
                             
                        elif self.view_2d_zoom > 1.0:
                            uv_w = 1.0 / self.view_2d_zoom
                            uv_h = 1.0 / self.view_2d_zoom
                            cx, cy = self.view_2d_center
                            # Re-calc to match exact logic above or use cached?
                            # Let's re-calc to be safe and stateless here
                            _sx = int((cx - uv_w / 2) * w_full)
                            _sy = int((cy - uv_h / 2) * h_full)
                            # Manual zoom clamps top-left to 0
                            src_x = max(0, _sx)
                            src_y = max(0, _sy)
                             
                            src_w = img.shape[1] 
                            src_h = img.shape[0] 
                         
                        target_w = scaled.width()
                        target_h = scaled.height()
                        
                        sx_factor = target_w / src_w if src_w > 0 else 0
                        sy_factor = target_h / src_h if src_h > 0 else 0
                        
                        for (ox, oy, orad, ocol, is_head) in points_to_draw:
                            # Project to screen
                            screen_x = (ox - src_x) * sx_factor
                            screen_y = (oy - src_y) * sy_factor
                            
                            # Skip if out of view
                            if screen_x < 0 or screen_x > target_w or screen_y < 0 or screen_y > target_h:
                                continue
                            
                            qcolor = QColor(ocol[0], ocol[1], ocol[2])
                            
                            if is_head:
                                # Head
                                painter.setPen(QPen(qcolor, 2)) # Thickness 2
                                painter.setBrush(Qt.BrushStyle.NoBrush)
                                
                                # Bubble vs Tracer Logic
                                if orad > 0:
                                    # Bubble: match actual radius in screen pixels (scales with zoom)
                                    draw_radius = orad * sx_factor
                                else:
                                    # Tracer: Fixed screen size (Radius 4.0)
                                    draw_radius = 4.0
                                    
                                painter.drawEllipse(QPointF(screen_x, screen_y), draw_radius, draw_radius)
                            else:
                                # Trail: Small solid dots
                                painter.setPen(Qt.PenStyle.NoPen)
                                painter.setBrush(qcolor)
                                painter.drawEllipse(QPointF(screen_x, screen_y), 2.0, 2.0)
                                
                        painter.end()
                        
                        self.cam_labels[i].setPixmap(scaled)
                    else:
                        self.cam_labels[i].setText(f"{cam_name} (pixmap null)")
                        
                except Exception as img_e:
                    self.cam_labels[i].setText(f"{cam_name} (Err: {str(img_e)[:15]})")
                    
        except Exception as e:
            print(f"2D Update Error: {e}")

    # --- 2D View Zoom/Pan Logic ---
    
    def _on_2d_zoom(self, delta):
        step = 0.1 * self.view_2d_zoom 
        if delta > 0:
            self.view_2d_zoom = min(10.0, self.view_2d_zoom + step)
        else:
            self.view_2d_zoom = max(1.0, self.view_2d_zoom - step)
        if self.view_2d_zoom == 1.0:
            self.view_2d_center = (0.5, 0.5)
        self._update_2d_view_frame() 
        
    def _on_2d_pan(self, norm_dx, norm_dy):
        if self.view_2d_zoom <= 1.0: return
        uv_w, uv_h = 1.0 / self.view_2d_zoom, 1.0 / self.view_2d_zoom
        shift_x, shift_y = norm_dx * uv_w, norm_dy * uv_h
        cx, cy = self.view_2d_center
        new_cx, new_cy = cx - shift_x, cy - shift_y
        half_w, half_h = uv_w / 2, uv_h / 2
        new_cx = max(half_w, min(1.0 - half_w, new_cx))
        new_cy = max(half_h, min(1.0 - half_h, new_cy))
        self.view_2d_center = (new_cx, new_cy)
        self._update_2d_view_frame()
        
    def _on_2d_reset(self):
        self.view_2d_zoom = 1.0
        self.view_2d_center = (0.5, 0.5)
        self._update_2d_view_frame()

    def _on_timeline_scrub(self, value):
        """Handle animation timeline scrubbing."""
        if not hasattr(self, 'anim_frame_range') or not self.anim_frame_range:
            return
            
        self.current_anim_frame = value
        
        # If animating, the loop handles redraw. If paused, force redraw.
        # Ideally, just redraw immediately for responsiveness.
        
        # Only update 3D if 3D View tab is active (index 0)
        current_vis_tab = self.vis_tabs.currentIndex()
        if current_vis_tab == 0 and self.current_ax:
             self._render_3d_frame()
        
        # Update 2D if 2D View tab is active (index 1)
        if current_vis_tab == 1:
            self._update_2d_view_frame()

    def _on_anim_enable_changed(self, state):
        """Handle 'Enable Animation' toggle."""
        is_enabled = (state == Qt.CheckState.Checked.value)
        self.anim_btn.setEnabled(is_enabled)
        
        if not is_enabled and self.is_animating:
            self._toggle_animation() # Stop if active
            
        # Re-plot (either switching to frame-1 mode or full-static mode)
        if self.cached_lengths and self.cached_coords:
            self._display_statistics(self.cached_lengths, self.cached_coords, force_redraw=True)

    def _update_histogram(self, lengths):
        """Update the length histogram plot."""
        # Clear previous widget
        for i in reversed(range(self.hist_layout.count())): 
            widget = self.hist_layout.itemAt(i).widget()
            if widget:
                widget.setParent(None)

        try:
            # Use explicit Figure object to avoid global plt state (prevents external windows)
            fig = Figure(figsize=(5, 2.2))
            fig.patch.set_facecolor('#000000') 
            ax = fig.add_subplot(111)
            ax.set_facecolor('#000000')
            
            # Use discrete bins for lengths
            bins = range(int(min(lengths)), int(max(lengths)) + 2) if lengths else 10
            ax.hist(lengths, bins=bins, color='#00d4ff', edgecolor='#00aacc', alpha=1.0)
            
            # No title, set labels
            ax.set_xlabel("Length (frames)", color='white', fontsize=9)
            ax.set_ylabel("Frequency", color='white', fontsize=9)
            
            # Adjust X-axis range to 95th percentile
            if len(lengths) > 0:
                p95 = np.percentile(lengths, 95)
                ax.set_xlim(0, max(p95 * 1.05, 10))
            
            ax.tick_params(colors='white', labelsize=8)
            for spine in ax.spines.values():
                spine.set_edgecolor('#444')
            
            # Maximize use of space, minimize bottom margin even further
            fig.subplots_adjust(left=0.15, right=0.98, top=0.95, bottom=0.22)
            
            canvas = FigureCanvas(fig)
            self.hist_layout.addWidget(canvas)
        except Exception as e:
            err_label = QLabel(f"Plot Error: {e}")
            err_label.setStyleSheet("color: #ff4444;")
            err_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
            self.hist_layout.addWidget(err_label)

    def _update_3d_view(self, track_coords):
        """Plot 3D trajectories in the left panel."""
        # Clear previous widget
        for i in reversed(range(self.vis_3d_layout.count())): 
            widget = self.vis_3d_layout.itemAt(i).widget()
            if widget: widget.setParent(None)

        try:
            from mpl_toolkits.mplot3d import Axes3D
            
            # Use explicit Figure object to avoid global plt state (prevents external windows)
            if not self.current_fig or not self.isVisible():
                self.current_fig = Figure(figsize=(8, 8))
                self.current_fig.patch.set_facecolor('#000000')
                self.current_ax = self.current_fig.add_subplot(111, projection='3d')
                self.current_ax.set_facecolor('#000000')
            else:
                self.current_ax.clear()

            fig = self.current_fig
            ax = self.current_ax
            
            # Reset facecolor after clear
            ax.set_facecolor('#000000')
            
            # Store initial limits for the Home button reset (only first time)
            if not hasattr(self, '_initial_xlim'):
                self._initial_xlim = ax.get_xlim3d()
                self._initial_ylim = ax.get_ylim3d()
                self._initial_zlim = ax.get_zlim3d()

            # Custom Scroll Zoom Handler (Axis-limit based for reliability)
            def on_scroll(event):
                if event.inaxes != ax: return
                base_scale = 1.25
                try:
                    cur_xlim = ax.get_xlim3d()
                    cur_ylim = ax.get_ylim3d()
                    cur_zlim = ax.get_zlim3d()
                    
                    x_mid = (cur_xlim[0] + cur_xlim[1]) / 2.0
                    y_mid = (cur_ylim[0] + cur_ylim[1]) / 2.0
                    z_mid = (cur_zlim[0] + cur_zlim[1]) / 2.0
                    
                    x_half = (cur_xlim[1] - cur_xlim[0]) / 2.0
                    y_half = (cur_ylim[1] - cur_ylim[0]) / 2.0
                    z_half = (cur_zlim[1] - cur_zlim[0]) / 2.0
                    
                    if event.button == 'up':
                        scale_factor = 1.0 / base_scale
                    elif event.button == 'down':
                        scale_factor = base_scale
                    else:
                        return
                        
                    ax.set_xlim3d([x_mid - x_half * scale_factor, x_mid + x_half * scale_factor])
                    ax.set_ylim3d([y_mid - y_half * scale_factor, y_mid + y_half * scale_factor])
                    ax.set_zlim3d([z_mid - z_half * scale_factor, z_mid + z_half * scale_factor])
                    
                    fig.canvas.draw_idle()
                except: pass
            
            fig.canvas.mpl_connect('scroll_event', on_scroll)
            fig.canvas._on_scroll = on_scroll

            # Coordinate format label
            self.coord_label = QLabel("X: 0.000, Y: 0.000, Z: 0.000")
            self.coord_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
            self.coord_label.setStyleSheet("color: #00ff00; font-family: 'Consolas'; font-size: 13px; margin-bottom: 2px;")

            def on_mouse_move(event):
                if event.inaxes != ax:
                    return
                try:
                    # Accessing the formatted coordinate string from ax
                    coord_str = ax.format_coord(event.xdata, event.ydata)
                    # Extract numbers using regex
                    parts = re.findall(r"[-+]?\d*\.\d+|\d+", coord_str)
                    if len(parts) >= 3:
                        # 4 Significant figures: 小数点前后共有 4 个数字
                        # Example: 10.5367 -> 10.54, 5.2541 -> 5.254, 0.2043 -> 0.2043
                        formatted = f"X: {float(parts[0]):.4g}, Y: {float(parts[1]):.4g}, Z: {float(parts[2]):.4g}"
                        self.coord_label.setText(formatted)
                except: pass

            fig.canvas.mpl_connect('motion_notify_event', on_mouse_move)

            # Faster plotting: flatten all coordinates into one scatter-style plot
            all_x, all_y, all_z = [], [], []
            
            for tid, pts in track_coords.items():
                pts = np.array(pts)
                # pts is now (N, 4) -> [frame, x, y, z]
                
                # Show all points (full trajectories)
                all_x.extend(pts[:, 1])
                all_y.extend(pts[:, 2])
                all_z.extend(pts[:, 3])

            if all_x:
                # Use '.' markers and blue color
                ax.plot(all_x, all_y, all_z, 'b.', markersize=0.5, alpha=0.5)
            
            # Aesthetic adjustments
            ax.set_xlabel('X (mm)', color='white', fontsize=10)
            ax.set_ylabel('Y (mm)', color='white', fontsize=10)
            ax.set_zlabel('Z (mm)', color='white', fontsize=10)
            
            ax.tick_params(colors='white', labelsize=8)
            ax.xaxis.pane.set_facecolor('#000000')
            ax.yaxis.pane.set_facecolor('#000000')
            ax.zaxis.pane.set_facecolor('#000000')
            ax.xaxis.pane.set_edgecolor('#333')
            ax.yaxis.pane.set_edgecolor('#333')
            ax.zaxis.pane.set_edgecolor('#333')
            
            # Set grid color
            ax.grid(True, color='#222')
            
            fig.tight_layout()
            
            if not self.current_canvas:
                self.current_canvas = FigureCanvas(fig)
            
            canvas = self.current_canvas
            
            # Add Navigation Toolbar (Larger, Centered, Premium Style)
            toolbar_container = QWidget()
            toolbar_container.setStyleSheet("background-color: #000; border: none;")
            tb_main_layout = QVBoxLayout(toolbar_container)
            tb_main_layout.setContentsMargins(0, 5, 0, 5)
            tb_main_layout.setSpacing(5)
            
            # Row 1: Toolbar
            tb_row1 = QHBoxLayout()
            tb_row1.addStretch()
            toolbar = NavigationToolbar(canvas, self.vis_3d_widget)
            toolbar.setIconSize(QSize(18, 18))
            toolbar.setStyleSheet("""
                QToolBar { background-color: transparent; border: none; spacing: 8px; }
                QToolButton { 
                    color: #fff; 
                    background-color: #1a1a1a; 
                    border: 1px solid #333; 
                    border-radius: 12px; 
                    padding: 4px 12px;
                }
                QToolButton:hover { background-color: #333; border-color: #00d4ff; }
                QToolButton:checked { background-color: #00d4ff; color: #000; }
            """)
            toolbar.setFixedHeight(40)
            
            # COMPLETELY HIDE default coordinate display in the toolbar
            toolbar.set_message = lambda s: None
            for child in toolbar.findChildren(QLabel):
                child.hide()

            # Override Home button to also reset our manual axis limits
            original_home = toolbar.home
            def custom_home():
                ax.set_xlim3d(self._initial_xlim)
                ax.set_ylim3d(self._initial_ylim)
                ax.set_zlim3d(self._initial_zlim)
                original_home()
            toolbar.home = custom_home

            tb_row1.addWidget(toolbar)
            tb_row1.addStretch()
            tb_main_layout.addLayout(tb_row1)
            
            # Row 2: Coordinates (4 Significant Figures)
            tb_main_layout.addWidget(self.coord_label)
            
            self.vis_3d_layout.addWidget(toolbar_container)
            self.vis_3d_layout.setSpacing(0) # Ensure no gap
            self.vis_3d_layout.addWidget(canvas)
            
            # Auto-switch to 3D tab if points were loaded
            self.vis_tabs.setCurrentIndex(0)
            
            # Explicitly trigger redraw
            canvas.draw()
            
        except Exception as e:
            err_label = QLabel(f"3D Plot Error: {e}")
            err_label.setStyleSheet("color: #ff4444;")
            err_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
            self.vis_3d_layout.addWidget(err_label)

    def _on_slider_moved(self, value):
        """Update live label during slider movement (fast)."""
        self.track_slider_label.setText(str(value))

    def _on_slider_released(self):
        """Trigger re-plot only when user finishes sliding (heavy)."""
        if self.cached_lengths and self.cached_coords:
            self._display_statistics(self.cached_lengths, self.cached_coords)
