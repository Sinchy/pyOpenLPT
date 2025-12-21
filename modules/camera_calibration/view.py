
import numpy as np

import matplotlib
matplotlib.use('Qt5Agg')
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure
from mpl_toolkits.mplot3d import Axes3D

from PySide6.QtWidgets import (QWidget, QVBoxLayout, QHBoxLayout, QLabel, 
                             QPushButton, QTabWidget, QComboBox, QSpinBox, 
                             QDoubleSpinBox, QListWidget, QGroupBox, QFormLayout,
                             QCheckBox, QFileDialog, QScrollArea, QFrame,
                             QTableWidget, QTableWidgetItem, QHeaderView, QMessageBox,
                             QSizePolicy)
from PySide6.QtCore import Qt
from .widgets import RangeSlider
from .wand_calibrator import WandCalibrator

class NumericTableWidgetItem(QTableWidgetItem):
    """TableWidgetItem that sorts numerically instead of alphabetically."""
    def __lt__(self, other):
        try:
            return float(self.text()) < float(other.text())
        except ValueError:
            return super().__lt__(other)

class Calibration3DViewer(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        try:
            self.figure = Figure(figsize=(5, 5), dpi=100)
            self.figure.patch.set_facecolor('black') # Dark theme
            self.canvas = FigureCanvas(self.figure)
            
            layout = QVBoxLayout(self)
            layout.setContentsMargins(0,0,0,0)
            layout.addWidget(self.canvas)
            
            self.ax = self.figure.add_subplot(111, projection='3d')
            self.reset_plot()
            self._initialized = True
            print("[Calibration3DViewer] Initialized successfully")
        except Exception as e:
            print(f"[Calibration3DViewer] ERROR during init: {e}")
            import traceback
            traceback.print_exc()
            self._initialized = False
            # Fallback: show error label
            layout = QVBoxLayout(self)
            layout.addWidget(QLabel(f"3D Viewer Error: {e}"))
        
    def reset_plot(self):
        self.ax.clear()
        self.ax.set_facecolor('black')
        self.ax.tick_params(colors='white')
        self.ax.xaxis.label.set_color('white')
        self.ax.yaxis.label.set_color('white')
        self.ax.zaxis.label.set_color('white')
        self.ax.set_xlabel('X (m)')
        self.ax.set_ylabel('Y (m)')
        self.ax.set_zlabel('Z (m)')
        
        # Keep pane borders but remove fill
        self.ax.xaxis.pane.fill = False
        self.ax.yaxis.pane.fill = False
        self.ax.zaxis.pane.fill = False
        self.ax.xaxis.pane.set_edgecolor('gray')
        self.ax.yaxis.pane.set_edgecolor('gray')
        self.ax.zaxis.pane.set_edgecolor('gray')
        
        # Equal aspect
        self.ax.set_box_aspect([1, 1, 1])
        
        self.canvas.draw()
        
    def plot_calibration(self, cameras, points_3d=None):
        """Plot cameras and points in 3D (MATLAB style)."""
        self.ax.clear()
        
        # Re-apply dark theme styles
        self.ax.set_facecolor('black')
        self.ax.tick_params(colors='white')
        self.ax.xaxis.label.set_color('white')
        self.ax.yaxis.label.set_color('white')
        self.ax.zaxis.label.set_color('white')
        self.ax.set_xlabel('X (m)')
        self.ax.set_ylabel('Y (m)')
        self.ax.set_zlabel('Z (m)')
        
        # Keep pane borders but remove fill
        self.ax.xaxis.pane.fill = False
        self.ax.yaxis.pane.fill = False
        self.ax.zaxis.pane.fill = False
        self.ax.xaxis.pane.set_edgecolor('gray')
        self.ax.yaxis.pane.set_edgecolor('gray')
        self.ax.zaxis.pane.set_edgecolor('gray')
        
        # Convert mm to m (divide by 1000)
        scale = 1000.0
        
        all_x, all_y, all_z = [], [], []
        
        # 1. Plot Wand Points (Blue)
        if points_3d is not None and len(points_3d) > 0:
            xs = points_3d[:, 0] / scale
            ys = points_3d[:, 1] / scale
            zs = points_3d[:, 2] / scale
            self.ax.scatter(xs, ys, zs, c='#0077FF', s=5, marker='.', alpha=0.7, label='Wand Points')
            all_x.extend(xs)
            all_y.extend(ys)
            all_z.extend(zs)
        
        # Collect camera positions first
        camera_data = []
        if cameras:
            for c_id, params in cameras.items():
                if 'R' in params and 'T' in params:
                    R = params['R']
                    t = params['T']
                    C = -R.T @ t
                    C = C.flatten() / scale
                    camera_data.append((c_id, C, R))
                    all_x.append(C[0])
                    all_y.append(C[1])
                    all_z.append(C[2])
        
        # True axis equal (like MATLAB's axis equal) - same physical scale
        max_range = 0.5  # default
        if all_x and all_y and all_z:
            max_range = max(max(all_x)-min(all_x), max(all_y)-min(all_y), max(all_z)-min(all_z)) / 2.0
            mid_x = (max(all_x) + min(all_x)) / 2.0
            mid_y = (max(all_y) + min(all_y)) / 2.0
            mid_z = (max(all_z) + min(all_z)) / 2.0
            self.ax.set_xlim(mid_x - max_range, mid_x + max_range)
            self.ax.set_ylim(mid_y - max_range, mid_y + max_range)
            self.ax.set_zlim(mid_z - max_range, mid_z + max_range)
        
        # Now draw cameras with direction lines (length = half of max_range)
        axis_len_m = max_range / 2.0
        for c_id, C, R in camera_data:
            # Plot Camera Center
            self.ax.scatter(C[0], C[1], C[2], c='black', s=100, marker='s', edgecolors='white', linewidths=2)
            
            # Text Label
            self.ax.text(C[0] + 0.02, C[1] + 0.02, C[2] + 0.02, f'Camera {c_id}', 
                        color='white', fontsize=10, fontweight='bold')
            
            # Direction line (length = half of range)
            z_dir = R.T @ np.array([0, 0, 1]) * axis_len_m
            end_pt = C + z_dir
            self.ax.plot3D([C[0], end_pt[0]], [C[1], end_pt[1]], [C[2], end_pt[2]], 
                           color='#FFFF00', linewidth=2)
        
        self.ax.set_box_aspect([1, 1, 1])
        
        self.canvas.draw()

class CameraCalibrationView(QWidget):
# ... (skip to create_wand_tab_v2)

    def create_wand_tab_v2(self):
        """Create the Wand Calibration tab (Multi-Camera) - Tabbed Interface."""
        tab = QWidget()
        layout = QHBoxLayout(tab)
        
        # 1. Visualization (LEFT)
        vis_frame = QFrame()
        vis_frame.setStyleSheet("background-color: #000000; border: 1px solid #333;")
        vis_layout = QVBoxLayout(vis_frame)
        vis_layout.setContentsMargins(0,0,0,0)

        self.vis_tabs = QTabWidget()
        self.vis_tabs.setStyleSheet("""
            QTabWidget::pane { border: 0; }
            QTabBar::tab { background: #222; color: #888; padding: 5px; }
            QTabBar::tab:selected { background: #444; color: #fff; }
        """)
        
        # Tab 1: 2D Detection Images
        vis_2d_widget = QWidget()
        vis_2d_layout = QVBoxLayout(vis_2d_widget)
        vis_2d_layout.setContentsMargins(0,0,0,0)
        
        # Scroll Area for images
        vis_scroll = QScrollArea()
        vis_scroll.setWidgetResizable(True)
        vis_scroll_content = QWidget()
        self.vis_grid_layout = QFormLayout(vis_scroll_content) # Or Grid? Form is Vertical.
        vis_scroll.setWidget(vis_scroll_content)
        vis_2d_layout.addWidget(vis_scroll)
        
        self.vis_tabs.addTab(vis_2d_widget, "2D Detection")
        
        # Tab 2: 3D Visualization
        self.calib_3d_view = Calibration3DViewer()
        self.vis_tabs.addTab(self.calib_3d_view, "3D View")
        
        vis_layout.addWidget(self.vis_tabs)
        
        self.cam_vis_labels = {} # Will be populated dynamically in _update_wand_table? 
        # Or I should add initial labels here?
        # Let's keep it empty and rely on dynamic update if exists, or just leave as is.
        # But wait, `_detect_single_frame` uses `self.cam_vis_labels`.
        # I need to ensure they are created.
        # I'll create 4 default labels.
        for i in range(4):
            lbl = QLabel(f"Cam {i} (No Image)")
            lbl.setAlignment(Qt.AlignmentFlag.AlignCenter)
            lbl.setStyleSheet("background: #111; border: 1px dashed #333; color: #555;")
            lbl.setMinimumHeight(200)
            self.vis_grid_layout.addRow(f"Cam {i}:", lbl)
            self.cam_vis_labels[i] = lbl
        
        # 2. Controls Panel (RIGHT)
        right_panel = QWidget()
        right_panel.setFixedWidth(370)
        right_layout = QVBoxLayout(right_panel)
        right_layout.setContentsMargins(0, 0, 0, 0)
        
        # Style for Tabs - Black Background
        controls_tabs = QTabWidget()
        controls_tabs.setStyleSheet("""
             QTabWidget::pane { border: 1px solid #444; background: #000000; }
             QTabBar::tab { background: #333; color: #aaa; padding: 8px; min-width: 100px; }
             QTabBar::tab:selected { background: #444; color: #fff; border-bottom: 2px solid #00d4ff; }
        """)

        # Common Button Style
        btn_style = """
            QPushButton {
                background-color: #2a3f5f; 
                color: white; 
                border: 1px solid #444; 
                border-radius: 4px; 
                padding: 8px;
                font-size: 13px;
                min-height: 25px;
            }
            QPushButton:hover { background-color: #3b5278; }
            QPushButton:pressed { background-color: #1e2d42; }
        """
        
        btn_style_primary = """
            QPushButton {
                background-color: #00d4ff; 
                color: black; 
                border: 1px solid #00a0cc; 
                border-radius: 4px; 
                padding: 8px;
                font-weight: bold;
                font-size: 13px;
                min-height: 25px;
            }
            QPushButton:hover { background-color: #66e5ff; }
            QPushButton:pressed { background-color: #008fb3; }
        """

        # --- Tab 1: Detection ---
        det_tab = QWidget()
        det_layout = QVBoxLayout(det_tab)
        det_layout.setSpacing(10)
        det_layout.setContentsMargins(10, 10, 10, 10)
        
        # Conf Group
        conf_group = QGroupBox("Detection Settings")
        conf_layout = QFormLayout(conf_group)
        
        self.wand_num_cams = QSpinBox()
        self._apply_input_style(self.wand_num_cams)
        self.wand_num_cams.setValue(4)
        self.wand_num_cams.setRange(1, 16)
        self.wand_num_cams.valueChanged.connect(self._update_wand_table)
        
        self.wand_type_combo = QComboBox()
        self._apply_input_style(self.wand_type_combo)
        self.wand_type_combo.addItems(["Dark on Bright", "Bright on Dark"])
        
        self.radius_range = RangeSlider(min_val=1, max_val=500, initial_min=20, initial_max=200, suffix=" px")
        
        from .widgets import SimpleSlider
        self.sensitivity_slider = SimpleSlider(min_val=0.5, max_val=1.0, initial=0.85, decimals=2)
        
        conf_layout.addRow("Num Cameras:", self.wand_num_cams)
        conf_layout.addRow("Wand Type:", self.wand_type_combo)
        conf_layout.addRow("Radius Range:", self.radius_range)
        conf_layout.addRow("Sensitivity:", self.sensitivity_slider)
        det_layout.addWidget(conf_group)
        
        # Table (with per-camera focal length and image size)
        det_layout.addWidget(QLabel("Camera Images:"))
        self.wand_cam_table = QTableWidget()
        self.wand_cam_table.setColumnCount(5)
        self.wand_cam_table.setHorizontalHeaderLabels(["", "Cam ID", "Focal (px)", "Width", "Height"])
        
        # Add Tooltip for Focal Length
        # Need to access the QTableWidgetItem for the header
        focal_header_item = self.wand_cam_table.horizontalHeaderItem(2)
        if focal_header_item:
            focal_header_item.setToolTip("Focal Length (px) = Focal Length (mm) / Sensor Pixel Size (mm)")

        header = self.wand_cam_table.horizontalHeader()
        header.setSectionResizeMode(0, QHeaderView.ResizeMode.ResizeToContents)
        header.setSectionResizeMode(1, QHeaderView.ResizeMode.ResizeToContents)
        header.setSectionResizeMode(2, QHeaderView.ResizeMode.Stretch)
        header.setSectionResizeMode(3, QHeaderView.ResizeMode.ResizeToContents)
        header.setSectionResizeMode(4, QHeaderView.ResizeMode.ResizeToContents)
        self.wand_cam_table.verticalHeader().setVisible(False)
        self.wand_cam_table.setSelectionMode(QTableWidget.SelectionMode.NoSelection)
        self.wand_cam_table.setShowGrid(True)
        self.wand_cam_table.setStyleSheet("background-color: transparent; border: 1px solid #333;")
        self.wand_cam_table.setMinimumHeight(130) 
        self._update_wand_table(4)
        det_layout.addWidget(self.wand_cam_table)
        
        # Frame List
        det_layout.addWidget(QLabel("Frame List (Click to Preview):"))
        self.frame_table = QTableWidget()
        self.frame_table.setColumnCount(2)
        self.frame_table.setHorizontalHeaderLabels(["Index", "Filename"])
        self.frame_table.horizontalHeader().setSectionResizeMode(1, QHeaderView.ResizeMode.Stretch)
        self.frame_table.verticalHeader().setVisible(False)
        self.frame_table.setSelectionBehavior(QTableWidget.SelectionBehavior.SelectRows)
        self.frame_table.setSelectionMode(QTableWidget.SelectionMode.SingleSelection)
        self.frame_table.setStyleSheet("background-color: transparent; border: 1px solid #333;")
        self.frame_table.cellClicked.connect(self._on_frame_table_clicked)
        self.frame_table.setFixedHeight(120) 
        det_layout.addWidget(self.frame_table)

        # Actions
        self.btn_detect_single = QPushButton("Test Detect (Current Frame)")
        self.btn_detect_single.setStyleSheet(btn_style)
        self.btn_detect_single.clicked.connect(self._detect_single_frame)
        det_layout.addWidget(self.btn_detect_single)

        self.btn_process_wand = QPushButton("Process All Frames / Resume")
        self.btn_process_wand.setStyleSheet(btn_style)
        self.btn_process_wand.clicked.connect(self._process_wand_frames)
        det_layout.addWidget(self.btn_process_wand)
        
        det_layout.addStretch()

        # --- Tab 2: Calibration (with scroll area) ---
        cal_tab = QWidget()
        cal_scroll = QScrollArea()
        cal_scroll.setWidgetResizable(True)
        cal_scroll.setFrameShape(QFrame.Shape.NoFrame)
        
        cal_content = QWidget()
        cal_layout = QVBoxLayout(cal_content)
        cal_layout.setSpacing(15)
        cal_layout.setContentsMargins(10, 10, 10, 10)
        
        cal_group = QGroupBox("Calibration Settings")
        cal_form = QFormLayout(cal_group)
        
        self.wand_model_combo = QComboBox()
        self._apply_input_style(self.wand_model_combo)
        self.wand_model_combo.addItems(["Pinhole", "Polynomial"])
        
        self.wand_len_spin = QDoubleSpinBox()
        self._apply_input_style(self.wand_len_spin)
        self.wand_len_spin.setValue(10.0)
        self.wand_len_spin.setRange(1, 5000)
        self.wand_len_spin.setSuffix(" mm")
        
        # Distortion Coefficient Count (0 = no distortion, up to 4)
        self.dist_coeff_spin = QSpinBox()
        self._apply_input_style(self.dist_coeff_spin)
        self.dist_coeff_spin.setValue(0)
        self.dist_coeff_spin.setRange(0, 4)
        self.dist_coeff_spin.setToolTip("Number of radial distortion coefficients (0=none, 1=k1, 2=k1+k2, etc.)")

        cal_form.addRow("Camera Model:", self.wand_model_combo)
        cal_form.addRow("Wand Length:", self.wand_len_spin)
        cal_form.addRow("Dist Coeffs:", self.dist_coeff_spin)
        cal_layout.addWidget(cal_group)
        
        cal_layout.addStretch()
        
        # Load Points Button
        self.btn_load_points = QPushButton("Load Wand Points (from CSV)")
        self.btn_load_points.setStyleSheet(btn_style)
        self.btn_load_points.clicked.connect(self._load_wand_points_for_calibration)
        cal_layout.addWidget(self.btn_load_points)
        
        # Precalibrate Check
        self.btn_precalibrate = QPushButton("Precalibrate to Check")
        self.btn_precalibrate.setStyleSheet("background-color: #ff9800; color: #000000; font-weight: bold; padding: 10px;")
        self.btn_precalibrate.clicked.connect(lambda: self._run_wand_calibration(precalibrate=True))
        cal_layout.addWidget(self.btn_precalibrate)
        
        # Run Calibration
        self.btn_calibrate_wand = QPushButton("Run Calibration")
        self.btn_calibrate_wand.setStyleSheet(btn_style_primary)
        self.btn_calibrate_wand.clicked.connect(self._run_wand_calibration)
        cal_layout.addWidget(self.btn_calibrate_wand)
        
        # --- Error Analysis Section (shown after calibration) ---
        error_header_row = QHBoxLayout()
        error_header_row.addWidget(QLabel("Error Analysis:"))
        
        # Warning label for missing image paths (inline, hidden by default)
        self.error_warning_label = QLabel("")
        self.error_warning_label.setStyleSheet("color: #ff6b6b; font-size: 11px;")
        self.error_warning_label.setVisible(False)
        error_header_row.addWidget(self.error_warning_label)

        
        error_header_row.addStretch()
        cal_layout.addLayout(error_header_row)
        
        # Error Table with horizontal scroll and sorting
        # Error Table with Frozen "Remove" Column
        error_table_container = QWidget()
        error_table_layout = QHBoxLayout(error_table_container)
        error_table_layout.setContentsMargins(0, 0, 0, 0)
        error_table_layout.setSpacing(0)
        
        # 1. Fixed "Frozen" Table (Left, Col 0 only)
        self.frozen_table = QTableWidget()
        self.frozen_table.setMinimumHeight(200)
        self.frozen_table.setStyleSheet("""
            QTableWidget { background-color: #1a1a2e; color: white; border: none; border-right: 1px solid #444; }
            QHeaderView::section { background-color: #2a2a4e; color: white; border: none; }
        """)
        self.frozen_table.setColumnCount(1)
        self.frozen_table.setHorizontalHeaderLabels(["Del"])
        self.frozen_table.setVerticalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff) # Scroll controlled by right table
        self.frozen_table.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        self.frozen_table.horizontalHeader().setSectionResizeMode(QHeaderView.ResizeMode.Fixed)
        self.frozen_table.setFixedWidth(42) # Compact width for checkbox
        self.frozen_table.setColumnWidth(0, 42)
        self.frozen_table.verticalHeader().setVisible(False)
        self.frozen_table.setSelectionMode(QTableWidget.SelectionMode.NoSelection)
        
        # 2. Main Scrollable Table (Right, Cols 1..N)
        self.error_table = QTableWidget()
        self.error_table.setMinimumHeight(200)
        self.error_table.setStyleSheet("""
            QTableWidget { background-color: #1a1a2e; color: white; border: none; }
            QHeaderView::section { background-color: #2a2a4e; color: white; border: none; }
        """)
        self.error_table.setHorizontalScrollMode(QTableWidget.ScrollMode.ScrollPerPixel)
        self.error_table.verticalHeader().setVisible(False)
        self.error_table.setSelectionMode(QTableWidget.SelectionMode.SingleSelection)
        self.error_table.cellClicked.connect(self._on_error_table_clicked)
        
        # Sync Vertical Scrollbars
        self.error_table.verticalScrollBar().valueChanged.connect(self.frozen_table.verticalScrollBar().setValue)
        self.frozen_table.verticalScrollBar().valueChanged.connect(self.error_table.verticalScrollBar().setValue)
        
        # Sync Sorting: when error_table is sorted, reorder frozen_table to match
        self.error_table.horizontalHeader().sortIndicatorChanged.connect(self._sync_frozen_table_sort)
        
        error_table_layout.addWidget(self.frozen_table)
        error_table_layout.addWidget(self.error_table)
        cal_layout.addWidget(error_table_container)
        
        # Batch Filter Controls (auto-update when changed)
        filter_row1 = QHBoxLayout()
        self.filter_proj_check = QCheckBox("Delete when proj error >")
        self.filter_proj_check.toggled.connect(self._auto_update_filter_marks)
        self.filter_proj_spin = QDoubleSpinBox()
        self.filter_proj_spin.setRange(0.1, 100)
        self.filter_proj_spin.setValue(5.0)
        self.filter_proj_spin.setSuffix(" px")
        self.filter_proj_spin.valueChanged.connect(self._auto_update_filter_marks)
        self._apply_input_style(self.filter_proj_spin)
        filter_row1.addWidget(self.filter_proj_check)
        filter_row1.addWidget(self.filter_proj_spin)
        filter_row1.addStretch()
        cal_layout.addLayout(filter_row1)
        
        filter_row2 = QHBoxLayout()
        self.filter_len_check = QCheckBox("Delete when wand len error >")
        self.filter_len_check.toggled.connect(self._auto_update_filter_marks)
        self.filter_len_spin = QDoubleSpinBox()
        self.filter_len_spin.setRange(0.01, 100)
        self.filter_len_spin.setValue(1.0)
        self.filter_len_spin.setSuffix(" mm")
        self.filter_len_spin.valueChanged.connect(self._auto_update_filter_marks)
        self._apply_input_style(self.filter_len_spin)
        filter_row2.addWidget(self.filter_len_check)
        filter_row2.addWidget(self.filter_len_spin)
        filter_row2.addStretch()
        cal_layout.addLayout(filter_row2)

        # Save Button (Relocated to bottom)
        save_row = QHBoxLayout()
        save_row.addStretch()
        self.btn_save_points = QPushButton("Save Filtered Points")
        # Make it slightly more prominent
        self.btn_save_points.setStyleSheet("background-color: #2a3f5f; color: white; font-size: 12px; padding: 6px 20px; border-radius: 4px;")
        self.btn_save_points.clicked.connect(self._save_filtered_points)
        save_row.addWidget(self.btn_save_points)
        cal_layout.addLayout(save_row)

        
        cal_layout.addStretch()
        
        # Setup scroll area and cal_tab layout
        cal_scroll.setWidget(cal_content)
        cal_tab_layout = QVBoxLayout(cal_tab)
        cal_tab_layout.setContentsMargins(0, 0, 0, 0)
        cal_tab_layout.addWidget(cal_scroll)

        # Add tabs
        controls_tabs.addTab(det_tab, "Point Detection")
        controls_tabs.addTab(cal_tab, "Calibration")
        
        # --- Tab 3: Tutorial ---
        tut_tab = QWidget()
        tut_layout = QVBoxLayout(tut_tab)
        tut_layout.setContentsMargins(20, 20, 20, 20)
        
        lbl_info = QLabel("Need help with Wand Calibration?")
        lbl_info.setStyleSheet("font-size: 14px; font-weight: bold; color: #00d4ff;")
        lbl_info.setAlignment(Qt.AlignmentFlag.AlignCenter)
        tut_layout.addWidget(lbl_info)
        
        lbl_desc = QLabel("Click the button below to open the comprehensive step-by-step user guide in your browser.")
        lbl_desc.setWordWrap(True)
        lbl_desc.setAlignment(Qt.AlignmentFlag.AlignCenter)
        tut_layout.addWidget(lbl_desc)
        
        tut_layout.addSpacing(20)
        
        self.btn_open_guide = QPushButton("Open User Guide")
        self.btn_open_guide.setStyleSheet(btn_style_primary)
        self.btn_open_guide.setCursor(Qt.CursorShape.PointingHandCursor)
        self.btn_open_guide.clicked.connect(self._open_user_guide)
        tut_layout.addWidget(self.btn_open_guide)
        
        tut_layout.addStretch()
        controls_tabs.addTab(tut_tab, "Tutorial")
        
        right_layout.addWidget(controls_tabs)
        
        # Progress Label
        self.status_label = QLabel("Status: Idle")
        self.status_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.status_label.setStyleSheet("color: #888; font-size: 11px; margin-top: 5px;")
        right_layout.addWidget(self.status_label)

        layout.addWidget(vis_frame, stretch=2)
        layout.addWidget(right_panel)
        
        return tab

    def _open_user_guide(self):
        """Open the HTML user guide in the default browser."""
        import os
        import webbrowser
        from pathlib import Path
        
        # Assume the HTML file is in the same directory as this script
        current_dir = Path(__file__).parent
        guide_path = current_dir / "WAND_CALIBRATION_USER_GUIDE.html"
        
        if guide_path.exists():
            webbrowser.open(guide_path.as_uri())
        else:
            QMessageBox.warning(self, "Guide Not Found", f"Could not find user guide at:\n{guide_path}")

    def _load_wand_points_for_calibration(self):
        """Prompt to load a CSV file, populate wand points, and ready for calibration."""
        file_path, _ = QFileDialog.getOpenFileName(self, "Load Wand Points", "", "CSV Files (*.csv)")
        if not file_path:
            return
            
        success, msg = self.wand_calibrator.load_wand_data_from_csv(file_path)
        if success:
            self.error_table.setRowCount(0)
            self._update_3d_viz()
            # Ensure calibrator has access to cameras/size
            if self.wand_images:
                 self.wand_calibrator.cameras = {}
                 for c, imgs in self.wand_images.items():
                     self.wand_calibrator.cameras[c] = {'images': imgs}
                     
            QMessageBox.information(self, "Success", msg + "\nand make sure image size is inputed.")
            count_frames = len(self.wand_calibrator.wand_points)
            self.status_label.setText(f"Loaded {count_frames} frames. Ready to calibrate.")
        else:
            QMessageBox.critical(self, "Error", f"Failed to load points:\n{msg}")
    def __init__(self):
        super().__init__()
        self.setup_ui()

    def setup_ui(self):
        # Initialize data structures
        self.plate_images = [] # List of absolute paths
        self.wand_images = {}  # Dict {cam_idx: [paths]}
        self.wand_calibrator = WandCalibrator() # Init calibrator

        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(10)

        # Title
        title = QLabel("Camera Calibration")
        title.setStyleSheet("font-size: 24px; font-weight: bold; color: #00d4ff; margin-bottom: 10px;")
        layout.addWidget(title)
        
        # Force input field styles for this view (Style fix)
        self.setStyleSheet("""
             QComboBox::drop-down { border: none; }
        """)

        # Main Tabs
        self.tabs = QTabWidget()
        self.tabs.addTab(self.create_plate_tab(), "Plate Calibration")
        self.tabs.addTab(self.create_wand_tab_v2(), "Wand Calibration")
        layout.addWidget(self.tabs)

    def _apply_input_style(self, widget):
        """Force style on input widgets to ensure background color."""
        # Note: We apply this to specific widgets to override any parent transparency issues
        widget.setStyleSheet("""
            background-color: #2d3a4a;
            border: 1px solid #3d4a5a;
            border-radius: 6px;
            padding: 6px 10px;
            color: #eaeaea;
            selection-background-color: #0f3460;
            max-width: 140px;
        """)
        if isinstance(widget, (QSpinBox, QDoubleSpinBox)):
            widget.setStyleSheet(widget.styleSheet() + """
                QSpinBox::up-button, QSpinBox::down-button,
                QDoubleSpinBox::up-button, QDoubleSpinBox::down-button {
                    width: 0px; height: 0px;
                }
            """)
        if isinstance(widget, QComboBox):
             widget.setStyleSheet(widget.styleSheet() + """
                QComboBox::drop-down { border: none; }
             """)

    def create_plate_tab(self):
        """Create the Plate Calibration tab (Single Camera)."""
        tab = QWidget()
        layout = QHBoxLayout(tab)
        
        # 1. Visualization (LEFT)
        vis_frame = QFrame()
        vis_frame.setStyleSheet("background-color: #000000; border: 1px solid #333;")
        vis_layout = QVBoxLayout(vis_frame)
        self.plate_vis_label = QLabel("No Image Loaded")
        self.plate_vis_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.plate_vis_label.setStyleSheet("color: #666;")
        vis_layout.addWidget(self.plate_vis_label)
        layout.addWidget(vis_frame, stretch=2)

        # 2. Controls (RIGHT)
        controls = QWidget()
        controls.setFixedWidth(350)
        controls_layout = QVBoxLayout(controls)
        controls_layout.setContentsMargins(10, 0, 10, 0)
        controls_layout.setSpacing(15)
        
        controls_scroll = QScrollArea()
        controls_scroll.setWidgetResizable(True)
        controls_scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        controls_scroll.setFixedWidth(370) 
        
        # -- Configuration Group --
        conf_group = QGroupBox("Configuration")
        conf_layout = QFormLayout(conf_group)
        
        self.num_cams_spin = QSpinBox()
        self._apply_input_style(self.num_cams_spin)
        self.num_cams_spin.setValue(4)
        self.num_cams_spin.setRange(1, 16)
        self.num_cams_spin.valueChanged.connect(self._update_cam_list)
        
        self.plate_cam_combo = QComboBox()
        self._apply_input_style(self.plate_cam_combo)
        
        self.plate_model_combo = QComboBox()
        self._apply_input_style(self.plate_model_combo)
        self.plate_model_combo.addItems(["Pinhole", "Polynomial"])
        
        conf_layout.addRow("Num Cameras:", self.num_cams_spin)
        conf_layout.addRow("Target Camera:", self.plate_cam_combo)
        conf_layout.addRow("Camera Model:", self.plate_model_combo)
        
        self.btn_load_plate = QPushButton("Load Images")
        self.btn_load_plate.setStyleSheet("background-color: #2a3f5f;")
        self.btn_load_plate.clicked.connect(self._load_plate_images)
        conf_layout.addRow(self.btn_load_plate)

        self.btn_clear_plate = QPushButton("Clear Images")
        self.btn_clear_plate.setStyleSheet("background-color: #3a3f4f; font-size: 11px;")
        self.btn_clear_plate.clicked.connect(self._clear_plate_images)
        conf_layout.addRow(self.btn_clear_plate)
        
        controls_layout.addWidget(conf_group)

        self._update_cam_list(4)

        # -- Plate Settings Group --
        plate_group = QGroupBox("Plate Settings")
        plate_layout = QFormLayout(plate_group)
        
        self.rows_spin = QSpinBox()
        self._apply_input_style(self.rows_spin)
        self.rows_spin.setValue(10)
        self.cols_spin = QSpinBox()
        self._apply_input_style(self.cols_spin)
        self.cols_spin.setValue(10)
        self.space_spin = QDoubleSpinBox()
        self._apply_input_style(self.space_spin)
        self.space_spin.setValue(10.0)
        self.space_spin.setSuffix(" mm")
        
        plate_layout.addRow("Rows:", self.rows_spin)
        plate_layout.addRow("Columns:", self.cols_spin)
        plate_layout.addRow("Spacing:", self.space_spin)
        controls_layout.addWidget(plate_group)
        
        controls_layout.addWidget(QLabel("Loaded Images:"))
        self.plate_img_list = QListWidget()
        self.plate_img_list.setMaximumHeight(100)
        self.plate_img_list.currentRowChanged.connect(self._display_plate_image)
        controls_layout.addWidget(self.plate_img_list)

        # -- Actions Group --
        action_layout = QVBoxLayout()
        self.btn_detect_plate = QPushButton("Detect Points")
        self.btn_detect_plate.setStyleSheet("background-color: #2a3f5f;")
        self.btn_detect_plate.clicked.connect(self._detect_plate_points)
        
        self.btn_calibrate_plate = QPushButton("Run Calibration")
        self.btn_calibrate_plate.setStyleSheet("background-color: #00d4ff; color: #000000; font-weight: bold;")
        self.btn_calibrate_plate.clicked.connect(self._run_plate_calibration)
        
        action_layout.addWidget(self.btn_detect_plate)
        action_layout.addWidget(self.btn_calibrate_plate)
        controls_layout.addLayout(action_layout)
        
        controls_layout.addStretch()
        controls_scroll.setWidget(controls)
        layout.addWidget(controls_scroll)
        
        return tab

    def create_wand_tab(self):
        """Create the Wand Calibration tab (Multi-Camera)."""
        tab = QWidget()
        layout = QHBoxLayout(tab)
        
        # 1. Visualization (LEFT)
        vis_frame = QFrame()
        vis_frame.setStyleSheet("background-color: #000000; border: 1px solid #333;")
        vis_layout = QVBoxLayout(vis_frame)
        vis_layout.setContentsMargins(0,0,0,0)

        # TabWidget for Cameras or Grid? 
        # User requested: "give each camera load first image"
        # Let's use a TabWidget because a Grid of 4 images at full Res inside a small frame is hard to see.
        # But user also said "visualization there give each camera load first image" which might imply simultaneous view.
        # Let's simple TabWidget for now, with "Camera 1", "Camera 2" tabs.
        
        self.vis_tabs = QTabWidget()
        self.vis_tabs.setStyleSheet("""
            QTabWidget::pane { border: 0; }
            QTabBar::tab { background: #222; color: #888; padding: 5px; }
            QTabBar::tab:selected { background: #444; color: #fff; }
        """)
        vis_layout.addWidget(self.vis_tabs)
        
        # We will populate tabs dynamically or pre-create for max cams?
        # Let's pre-create labels and store them
        self.cam_vis_labels = {} # {cam_idx: QLabel}
        
        # 2. Controls (RIGHT)
        controls = QWidget()
        controls.setFixedWidth(350)
        controls_layout = QVBoxLayout(controls)
        controls_layout.setContentsMargins(10, 0, 10, 0)
        controls_layout.setSpacing(15)
        
        controls_scroll = QScrollArea()
        controls_scroll.setWidgetResizable(True)
        controls_scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        controls_scroll.setFixedWidth(370)

        # -- Configuration --
        conf_group = QGroupBox("Configuration")
        conf_layout = QFormLayout(conf_group)
        
        self.wand_num_cams = QSpinBox()
        self._apply_input_style(self.wand_num_cams)
        self.wand_num_cams.setValue(4)
        self.wand_num_cams.setRange(1, 16)
        self.wand_num_cams.valueChanged.connect(self._update_wand_table)
        
        self.wand_model_combo = QComboBox()
        self._apply_input_style(self.wand_model_combo)
        self.wand_model_combo.addItems(["Pinhole", "Polynomial"])
        
        self.wand_len_spin = QDoubleSpinBox()
        self._apply_input_style(self.wand_len_spin)
        self.wand_len_spin.setValue(10.0)
        self.wand_len_spin.setSuffix(" mm")
        
        # New: Wand Type (Bright/Dark)
        self.wand_type_combo = QComboBox()
        self._apply_input_style(self.wand_type_combo)
        self.wand_type_combo.addItems(["Dark on Bright", "Bright on Dark"])
        
        # New: Circle Radius Range for detection (Range Slider)
        self.radius_range = RangeSlider(min_val=1, max_val=500, initial_min=20, initial_max=200, suffix=" px")
        
        # New: Sensitivity slider for detection
        from .widgets import SimpleSlider
        self.sensitivity_slider = SimpleSlider(min_val=0.5, max_val=1.0, initial=0.85, decimals=2)
        
        conf_layout.addRow("Num Cameras:", self.wand_num_cams)
        conf_layout.addRow("Camera Model:", self.wand_model_combo)
        conf_layout.addRow("Wand Length:", self.wand_len_spin)
        conf_layout.addRow("Wand Type:", self.wand_type_combo)
        conf_layout.addRow("Radius Range:", self.radius_range)
        conf_layout.addRow("Sensitivity:", self.sensitivity_slider)

        controls_layout.addWidget(conf_group)

        # -- Camera Images Table --
        controls_layout.addWidget(QLabel("Camera Images:"))
        
        self.wand_cam_table = QTableWidget()
        self.wand_cam_table.setColumnCount(3)
        self.wand_cam_table.setHorizontalHeaderLabels(["", "Camera", "Source"])
        
        header = self.wand_cam_table.horizontalHeader()
        header.setSectionResizeMode(0, QHeaderView.ResizeMode.ResizeToContents)
        header.setSectionResizeMode(1, QHeaderView.ResizeMode.Stretch)
        header.setSectionResizeMode(2, QHeaderView.ResizeMode.ResizeToContents)
        
        self.wand_cam_table.verticalHeader().setVisible(False)
        self.wand_cam_table.setSelectionMode(QTableWidget.SelectionMode.NoSelection)
        self.wand_cam_table.setShowGrid(False)
        self.wand_cam_table.setStyleSheet("background-color: transparent; border: 1px solid #333;")
        self.wand_cam_table.setFixedHeight(120) 
        
        self._update_wand_table(4)
        controls_layout.addWidget(self.wand_cam_table)

        # -- Frame Navigation (Directly in layout) --
        controls_layout.addWidget(QLabel("Frame List:"))
        
        self.frame_table = QTableWidget()
        self.frame_table.setColumnCount(2)
        self.frame_table.setHorizontalHeaderLabels(["Index", "Filename"])
        self.frame_table.horizontalHeader().setSectionResizeMode(1, QHeaderView.ResizeMode.Stretch)
        self.frame_table.verticalHeader().setVisible(False)
        self.frame_table.setSelectionBehavior(QTableWidget.SelectionBehavior.SelectRows)
        self.frame_table.setSelectionMode(QTableWidget.SelectionMode.SingleSelection)
        self.frame_table.setStyleSheet("background-color: transparent; border: 1px solid #333;")
        self.frame_table.cellClicked.connect(self._on_frame_table_clicked)
        self.frame_table.setFixedHeight(150) 
        
        controls_layout.addWidget(self.frame_table)

        # -- Actions --
        action_layout = QVBoxLayout()
        self.btn_detect_single = QPushButton("Detect Points (Current Frame)")
        self.btn_detect_single.setStyleSheet("background-color: #2a3f5f;")
        self.btn_detect_single.clicked.connect(self._detect_single_frame)

        self.btn_process_wand = QPushButton("1. Process All Frames")
        self.btn_process_wand.setStyleSheet("background-color: #2a3f5f;")
        self.btn_process_wand.clicked.connect(self._process_wand_frames)
        
        self.btn_calibrate_wand = QPushButton("2. Run Calibration")
        self.btn_calibrate_wand.setStyleSheet("background-color: #00d4ff; color: #000000; font-weight: bold;")
        self.btn_calibrate_wand.clicked.connect(self._run_wand_calibration)

        action_layout.addWidget(self.btn_detect_single)
        action_layout.addWidget(self.btn_process_wand)
        action_layout.addWidget(self.btn_calibrate_wand)
        
        # Progress Label
        self.status_label = QLabel("Status: Idle")
        self.status_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        action_layout.addWidget(self.status_label)

        controls_layout.addLayout(action_layout)

        controls_layout.addStretch()
        controls_scroll.setWidget(controls)
        layout.addWidget(vis_frame, stretch=2)
        layout.addWidget(controls_scroll)
        
        return tab

    def _update_cam_list(self, count):
        """Update camera dropdown based on count."""
        current_idx = self.plate_cam_combo.currentIndex()
        self.plate_cam_combo.clear()
        items = [f"Camera {i+1}" for i in range(count)]
        self.plate_cam_combo.addItems(items)
        if current_idx >= 0 and current_idx < count:
            self.plate_cam_combo.setCurrentIndex(current_idx)

    def _update_wand_table(self, count):
        self.wand_cam_table.setRowCount(count)
        self.wand_images = {i: [] for i in range(count)}
        
        # Update Vis Tabs
        try:
            self.vis_tabs.clear()
            self.cam_vis_labels = {}
            for i in range(count):
                lbl = QLabel(f"Cam {i+1} Image")
                lbl.setAlignment(Qt.AlignmentFlag.AlignCenter)
                lbl.setStyleSheet("background: #000;")
                self.cam_vis_labels[i] = lbl
                self.vis_tabs.addTab(lbl, f"Cam {i+1}")
            
            # Add 3D View tab at the end
            if not hasattr(self, 'calib_3d_view') or self.calib_3d_view is None:
                self.calib_3d_view = Calibration3DViewer()
            self.vis_tabs.addTab(self.calib_3d_view, "3D View")
            
        except RuntimeError:
            return # Widget deleted or not ready

        for i in range(count):
            # Col 0: Load Folder button (compact)
            btn = QPushButton("Load")
            btn.setStyleSheet("background-color: #2a3f5f; padding: 2px 6px; font-size: 10px;")
            btn.clicked.connect(lambda checked=False, idx=i: self._load_wand_folder_for_cam(idx))
            self.wand_cam_table.setCellWidget(i, 0, btn)
            
            # Col 1: Cam ID (read-only)
            name_item = QTableWidgetItem(f"{i}")
            name_item.setFlags(Qt.ItemFlag.NoItemFlags)
            name_item.setForeground(Qt.GlobalColor.white)
            self.wand_cam_table.setItem(i, 1, name_item)
            
            # Col 2: Focal Length (editable spinbox)
            focal_spin = QSpinBox()
            focal_spin.setRange(100, 1000000)
            focal_spin.setValue(9000)
            focal_spin.setStyleSheet("background: #222; color: white; border: none;")
            self.wand_cam_table.setCellWidget(i, 2, focal_spin)
            
            # Col 3: Width (editable spinbox)
            width_spin = QSpinBox()
            width_spin.setRange(1, 10000)
            width_spin.setValue(1280)
            width_spin.setStyleSheet("background: #222; color: white; border: none;")
            self.wand_cam_table.setCellWidget(i, 3, width_spin)
            
            # Col 4: Height (editable spinbox)
            height_spin = QSpinBox()
            height_spin.setRange(1, 10000)
            height_spin.setValue(800)
            height_spin.setStyleSheet("background: #222; color: white; border: none;")
            self.wand_cam_table.setCellWidget(i, 4, height_spin)

    # --- Logic Implementation ---

    def _load_plate_images(self, checked=False):
        files, _ = QFileDialog.getOpenFileNames(
            self, "Select Calibration Images", "", "Images (*.png *.jpg *.bmp *.tif)"
        )
        if files:
            self.plate_images.extend(files)
            for f in files:
                from pathlib import Path
                self.plate_img_list.addItem(Path(f).name)
            if self.plate_img_list.count() > 0:
                self.plate_img_list.setCurrentRow(self.plate_img_list.count() - 1)

    def _clear_plate_images(self, checked=False):
        self.plate_images.clear()
        self.plate_img_list.clear()
        self.plate_vis_label.clear()
        self.plate_vis_label.setText("No Images")

    def _display_plate_image(self, row):
        if row < 0 or row >= len(self.plate_images):
            return
        img_path = self.plate_images[row]
        from PySide6.QtGui import QPixmap
        pixmap = QPixmap(img_path)
        if not pixmap.isNull():
            scaled_pix = pixmap.scaled(
                self.plate_vis_label.size(), 
                Qt.AspectRatioMode.KeepAspectRatio, 
                Qt.TransformationMode.SmoothTransformation
            )
            self.plate_vis_label.setPixmap(scaled_pix)
        else:
            self.plate_vis_label.setText("Failed to load image")

    def _load_wand_folder_for_cam(self, cam_idx):
        folder = QFileDialog.getExistingDirectory(self, f"Select Image Folder for Camera {cam_idx+1}")
        if folder:
            from pathlib import Path
            p = Path(folder)
            # Find images
            files = sorted([str(f) for f in p.iterdir() if f.suffix.lower() in ['.png', '.jpg', '.bmp', '.tif', '.jpeg']])
            
            if files:
                self.wand_images[cam_idx] = files
                btn = self.wand_cam_table.cellWidget(cam_idx, 0)  # Column 0 now
                if btn:
                    btn.setText(f"{len(files)}")
                
                # Check consistency and update Frames Table
                self.populate_wand_table()
    
    def populate_wand_table(self):
        """Populates the frame table with filenames and status."""
        # Find max length and reference cam
        max_len = 0
        max_idx = -1
        for k, v in self.wand_images.items():
            if len(v) > max_len:
                max_len = len(v)
                max_idx = k
        
        self.frame_table.setRowCount(max_len)
        reference_files = self.wand_images.get(max_idx, [])
        
        from PySide6.QtGui import QFont
        
        for i in range(max_len):
            # Index
            idx_item = QTableWidgetItem(str(i+1))
            idx_item.setTextAlignment(Qt.AlignmentFlag.AlignCenter)
            self.frame_table.setItem(i, 0, idx_item)
            
            # Filename
            fname = "Frame " + str(i+1)
            if i < len(reference_files):
                from pathlib import Path
                fname = Path(reference_files[i]).name
            
            name_item = QTableWidgetItem(fname)
            self.frame_table.setItem(i, 1, name_item)
            
            # Check Status (Valid/Invalid)
            if hasattr(self, 'wand_calibrator') and \
               hasattr(self.wand_calibrator, 'wand_data_raw') and \
               i in self.wand_calibrator.wand_data_raw:
                
                # If frame processed but NOT in filtered, it's invalid
                if hasattr(self.wand_calibrator, 'wand_data_filtered') and \
                   i not in self.wand_calibrator.wand_data_filtered:
                    
                    # Mark as Invalid (Strikethrough + Red)
                    font = name_item.font()
                    font.setStrikeOut(True)
                    name_item.setFont(font)
                    name_item.setForeground(Qt.GlobalColor.red)
                    idx_item.setForeground(Qt.GlobalColor.red)
                else:
                    # Valid
                    name_item.setForeground(Qt.GlobalColor.white)
                    idx_item.setForeground(Qt.GlobalColor.white)
        
        # Select first row if valid
        if max_len > 0 and self.frame_table.currentRow() < 0:
             self.frame_table.selectRow(0)
             self._update_vis_frame(0)
                     


    def _on_frame_table_clicked(self, row, col):
        self._update_vis_frame(row)

    def _update_vis_frame(self, frame_idx):
        from PySide6.QtGui import QPixmap, QImage
        import cv2
        import numpy as np
        
        # Update labels for all cams
        for cam_idx, lbl in self.cam_vis_labels.items():
            if cam_idx in self.wand_images and frame_idx < len(self.wand_images[cam_idx]):
                 path = self.wand_images[cam_idx][frame_idx]
                 
                 # Use OpenCV to load (supports .tif and more formats)
                 img = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
                 if img is None:
                     lbl.setText("Image Load Error")
                     continue
                 
                 # Convert to RGB for Qt
                 if len(img.shape) == 2:  # Grayscale
                     img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
                 elif img.shape[2] == 4:  # RGBA
                     img = cv2.cvtColor(img, cv2.COLOR_BGRA2RGB)
                 else:  # BGR
                     img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                 
                 # Normalize to 8-bit if needed (for 16-bit TIF)
                 # Normalize to 8-bit if needed (for 16-bit TIF)
                 if img.dtype != np.uint8:
                     img = (img / img.max() * 255).astype(np.uint8)
                 
                 # Draw Points if available
                 # 1. Raw Data (Red Circles)
                 if hasattr(self, 'wand_calibrator') and \
                    hasattr(self.wand_calibrator, 'wand_data_raw') and \
                    frame_idx in self.wand_calibrator.wand_data_raw:
                    
                     ct_dict = self.wand_calibrator.wand_data_raw[frame_idx]
                     if cam_idx in ct_dict:
                         raw_pts = ct_dict[cam_idx]
                         # Draw Red
                         for p in raw_pts:
                             x, y, r = int(p[0]), int(p[1]), int(p[2])
                             cv2.circle(img, (x, y), r, (0, 0, 255), 2) # Red
                 
                 # 2. Filtered Data (Green/Cyan Circles)
                 if hasattr(self, 'wand_calibrator') and \
                    hasattr(self.wand_calibrator, 'wand_data_filtered') and \
                    frame_idx in self.wand_calibrator.wand_data_filtered:
                     
                     filt_dict = self.wand_calibrator.wand_data_filtered[frame_idx]
                     if cam_idx in filt_dict:
                         filt_pts = filt_dict[cam_idx] # [pt_small, pt_large]
                         
                         # Small point (Light Green)
                         p_s = filt_pts[0]
                         cv2.circle(img, (int(p_s[0]), int(p_s[1])), int(p_s[2]), (0, 255, 100), 2)
                         # Large point (Dark Green)
                         p_l = filt_pts[1]
                         cv2.circle(img, (int(p_l[0]), int(p_l[1])), int(p_l[2]), (0, 150, 50), 2)

                 h, w, ch = img.shape
                 bytes_per_line = ch * w
                 q_img = QImage(img.data, w, h, bytes_per_line, QImage.Format.Format_RGB888)
                 pix = QPixmap.fromImage(q_img)
                 
                 # Get label size, use default if too small
                 lbl_size = lbl.size()
                 if lbl_size.width() < 50 or lbl_size.height() < 50:
                     lbl_size = self.vis_tabs.size()
                 
                 scaled = pix.scaled(lbl_size, Qt.AspectRatioMode.KeepAspectRatio, Qt.TransformationMode.SmoothTransformation)
                 lbl.setPixmap(scaled)
            else:
                lbl.setText("No Image")

    def _detect_plate_points(self, checked=False):
        print("Detecting points...")

    def _run_plate_calibration(self, checked=False):
        print("Running Plate Calibration...")
    
    def _detect_single_frame(self, checked=False):
        # Run detection on current frame only and visualize (Async)
        from .wand_calibrator import WandCalibrator, WandDetectionSingleFrameWorker
        import cv2
        import numpy as np
        from PySide6.QtGui import QPixmap, QImage
        from PySide6.QtWidgets import QProgressDialog
        
        if not hasattr(self, 'wand_calibrator'):
             self.wand_calibrator = WandCalibrator()
             
        idx = self.frame_table.currentRow()
        if idx < 0:
            self.status_label.setText("Please select a frame first.")
            return

        wand_type = "dark" if "Dark" in self.wand_type_combo.currentText() else "bright"
        min_r, max_r = self.radius_range.value()
        sensitivity = self.sensitivity_slider.value()
        
        # Create dict for single frame
        single_frame_dict = {}
        for c, files in self.wand_images.items():
            if idx < len(files):
                 single_frame_dict[c] = files[idx] 
        
        if not single_frame_dict:
            return

        print(f"Detecting frame {idx}, mode='{wand_type}', radius=[{min_r},{max_r}], sensitivity={sensitivity}")
        
        # Worker Setup
        self._single_detect_worker = WandDetectionSingleFrameWorker(
            self.wand_calibrator, single_frame_dict, wand_type, min_r, max_r, sensitivity
        )
        self._single_detect_worker.finished_signal.connect(self._on_single_detection_finished)
        
        # Dialog Setup
        self._detect_dialog = QProgressDialog("Detecting points...", None, 0, 0, self)
        self._detect_dialog.setWindowTitle("Please Wait")
        self._detect_dialog.setWindowModality(Qt.WindowModality.WindowModal)
        self._detect_dialog.setMinimumDuration(0) # Show immediately
        self._detect_dialog.show()
        
        # Connect finish to close
        self._single_detect_worker.finished_signal.connect(self._detect_dialog.close)
        self._single_detect_worker.finished_signal.connect(self._single_detect_worker.deleteLater)
        
        self._single_detect_worker.start()

    def _on_single_detection_finished(self, res):
        """Handle async detection result."""
        # print(f"Detection results: {res}")
        
        import cv2
        import numpy as np
        from PySide6.QtGui import QPixmap, QImage
        
        idx = self.frame_table.currentRow()
        
        # Visualize result for each camera
        total_points = 0
        for cam_idx, pts in res.items():
            if cam_idx in self.cam_vis_labels:
                lbl = self.cam_vis_labels[cam_idx]
                if cam_idx in self.wand_images and idx < len(self.wand_images[cam_idx]):
                    path = self.wand_images[cam_idx][idx]
                    
                    # Load image with OpenCV
                    img = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
                    if img is None:
                        continue
                    
                    # Convert to RGB
                    if len(img.shape) == 2:
                        img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
                    elif img.shape[2] == 4:
                        img = cv2.cvtColor(img, cv2.COLOR_BGRA2RGB)
                    else:
                        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                    
                    # Normalize to 8-bit if needed
                    if img.dtype != np.uint8:
                        img = (img / img.max() * 255).astype(np.uint8)
                    
                    # Draw ALL detected circles with radius labels
                    if pts is not None and len(pts) > 0:
                        for pt in pts:
                            x, y = int(pt[0]), int(pt[1])
                            r = int(pt[2]) if len(pt) > 2 else 15
                            # Draw circle
                            cv2.circle(img, (x, y), r, (0, 100, 255), 3)
                            # Draw radius label next to circle
                            label = f"r={r}"
                            cv2.putText(img, label, (x + r + 5, y), 
                                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
                        total_points += len(pts)
                    
                    # Convert to QPixmap
                    h, w, ch = img.shape
                    bytes_per_line = ch * w
                    q_img = QImage(img.data, w, h, bytes_per_line, QImage.Format.Format_RGB888)
                    pix = QPixmap.fromImage(q_img)
                    
                    # Scale and Set
                    lbl_size = lbl.size()
                    if lbl_size.width() < 50 or lbl_size.height() < 50:
                        lbl_size = self.vis_tabs.size()
                    
                    scaled = pix.scaled(lbl_size, Qt.AspectRatioMode.KeepAspectRatio, Qt.TransformationMode.SmoothTransformation)
                    lbl.setPixmap(scaled)
        
        self.status_label.setText(f"Frame {idx}: Found {total_points} points in {len(res)} cameras.")

    def _process_wand_frames(self, checked=False):
        from .wand_calibrator import WandCalibrator, WandDetectionWorker
        from .widgets import ProcessingDialog
        from PySide6.QtWidgets import QMessageBox, QFileDialog, QDialog
        from PySide6.QtCore import QFileInfo
        
        if not hasattr(self, 'wand_calibrator'):
             self.wand_calibrator = WandCalibrator()
        
        # Check if we have images
        count = sum(len(imgs) for imgs in self.wand_images.values())
        if count == 0:
            QMessageBox.warning(self, "No Images", "Please load images first.")
            return

        # 0. Info Popup (UX Improvement)
        QMessageBox.information(self, "Process Frames", 
                                "Please select the path to save the detection results.\n\n"
                                "You can also select a previously-saved results file to RESUME processing.",
                                QMessageBox.StandardButton.Ok)

        # 1. Prompt for Save Path (Autosave)
        autosave_path, _ = QFileDialog.getSaveFileName(self, "Select Save File (Autosave)", "wand_points.csv", "CSV Files (*.csv)")
        if not autosave_path:
            return # User cancelled
            
        resume = False
        # 2. Check if file exists -> Resume?
        if QFileInfo(autosave_path).exists():
            # Ask user
            reply = QMessageBox.question(self, "Resume?", 
                                         f"File '{QFileInfo(autosave_path).fileName()}' exists.\nDo you want to RESUME processing from where it left off?\n\nYes: Resume (keep existing data)\nNo: Overwrite (restart)",
                                         QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No | QMessageBox.StandardButton.Cancel)
            
            if reply == QMessageBox.StandardButton.Cancel:
                return
            
            if reply == QMessageBox.StandardButton.Yes:
                resume = True
                # Data loading moved to Worker thread to prevent UI freeze
        
        # Get UI parameters
        wand_type = "dark" if "Dark" in self.wand_type_combo.currentText() else "bright"
        min_r, max_r = self.radius_range.value()
        sensitivity = self.sensitivity_slider.value()

        # Start Worker
        self._proc_thread = WandDetectionWorker(
            self.wand_calibrator, 
            self.wand_images, 
            wand_type, min_r, max_r, sensitivity,
            autosave_path=autosave_path,
            resume=resume
        )
        
        # Setup dialog
        self._proc_dialog = ProcessingDialog(self)
        self._proc_dialog.stop_signal.connect(self._proc_thread.stop)
        
        # Correctly handle Pause/Resume toggle
        self._proc_dialog.pause_signal.connect(lambda p: self._proc_thread.pause() if p else self._proc_thread.resume())
        # self._proc_dialog.resume_signal.connect(self._proc_thread.resume) # Removed invalid signal
        
        self._proc_thread.progress.connect(self._proc_dialog.update_progress)
        self._proc_thread.finished_signal.connect(self._on_process_finished)
        
        self._proc_thread.start()
        self._proc_dialog.exec() # Modal blocking
        
    def _on_process_finished(self, success, msg):
        self._proc_dialog.close()
        
        if success:
             self.status_label.setText(f"Done: {msg}")
             self.populate_wand_table()
             self.populate_wand_table()
             
             # Update Image Size UI if available (Note: these spinboxes are in the camera table now)
             # if hasattr(self.wand_calibrator, 'image_size') and self.wand_calibrator.image_size != (0,0):
             #     h, w = self.wand_calibrator.image_size
             #     # Image size is now read from camera table, not separate spinboxes
             
             # Data is auto-saved, no need to prompt export here
        else:
             self.status_label.setText(f"Status: {msg}")
             
             # If stopped by user, don't show error popup (Data already saved by worker)
             if "Stopped by user" in msg:
                 self.populate_wand_table()
                 return # Done
             
             # Check if we have partial data
             has_data = False
             if hasattr(self, 'wand_calibrator'):
                 if hasattr(self.wand_calibrator, 'wand_data_raw') and len(self.wand_calibrator.wand_data_raw) > 0:
                     has_data = True
                 elif hasattr(self.wand_calibrator, 'wand_points') and len(self.wand_calibrator.wand_points) > 0:
                     has_data = True
             
             if has_data:
                 from PySide6.QtWidgets import QMessageBox, QFileDialog
                 reply = QMessageBox.question(self, "Detection Failed", 
                                              f"Error: {msg}\n\nHowever, partial data was captured ({len(self.wand_calibrator.wand_data_raw)} frames). Do you want to export it?",
                                              QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No)
                 if reply == QMessageBox.StandardButton.Yes:
                     path, _ = QFileDialog.getSaveFileName(self, "Save Rescue Data", "wand_rescue.csv", "CSV Files (*.csv)")
                     if path:
                         self.wand_calibrator.export_wand_data(path)
             else:
                 from PySide6.QtWidgets import QMessageBox
                 QMessageBox.warning(self, "Detection Failed", msg)
        from PySide6.QtWidgets import QMessageBox, QFileDialog
        
        reply = QMessageBox.question(self, "Export Data", "Detection complete. Do you want to export the point data?", 
                                     QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No)
        
        if reply == QMessageBox.StandardButton.Yes:
             path, _ = QFileDialog.getSaveFileName(self, "Save Wand Data", "wand_points.csv", "CSV Files (*.csv)")
             if path:
                 success, msg = self.wand_calibrator.export_wand_data(path)
                 if success:
                     QMessageBox.information(self, "Export", "Data exported successfully.")
                 else:
                     QMessageBox.warning(self, "Export Failed", msg)
        
    def _apply_current_ui_filter(self):
        """Collect checked frames from table and apply to calibrator."""
        from PySide6.QtWidgets import QCheckBox
        
        # Initialize persistent set if not exists
        if not hasattr(self, '_removed_frames'):
            self._removed_frames = set()
        
        if not hasattr(self, 'frozen_table'):
             return

        # Read currently checked frames from UI
        currently_checked = set()
        for row in range(self.frozen_table.rowCount()):
            widget = self.frozen_table.cellWidget(row, 0)
            if widget:
                chk = widget.findChild(QCheckBox)
                if chk and chk.isChecked():
                    fid = chk.property('frame_id')
                    if fid is not None:
                         currently_checked.add(int(fid))
        
        # Accumulate: add newly checked frames to persistent set
        # (Frames already removed stay removed)
        self._removed_frames.update(currently_checked)
        
        # Apply accumulated removal
        self.wand_calibrator.reset_filter()
        if self._removed_frames:
             print(f"[Filter] UI: removing {len(self._removed_frames)} frames total: {sorted(list(self._removed_frames))}")
             remaining = self.wand_calibrator.apply_filter(self._removed_frames)
             print(f"[Filter] After apply_filter: {remaining} frames remaining in wand_points_filtered")
        else:
             print(f"[Filter] UI: no frames marked for removal, using all {len(self.wand_calibrator.wand_points)} frames")
            
    def _run_wand_calibration(self, checked=False, precalibrate=False):
        if not hasattr(self, 'wand_calibrator'):
             self.status_label.setText("Please detect points first.")
             return

        # Read per-camera settings from table
        num_cams = self.wand_cam_table.rowCount()
        camera_settings = {}  # {cam_id: {'focal': int, 'width': int, 'height': int}}
        
        for row in range(num_cams):
            # Read Cam ID from column 1
            id_item = self.wand_cam_table.item(row, 1)
            if id_item is None:
                continue
            try:
                cam_id = int(id_item.text())
            except:
                cam_id = row
            
            # Read focal length from column 2
            focal_spin = self.wand_cam_table.cellWidget(row, 2)
            focal = focal_spin.value() if focal_spin else 9000
            
            # Read width from column 3
            width_spin = self.wand_cam_table.cellWidget(row, 3)
            width = width_spin.value() if width_spin else 1280
            
            # Read height from column 4
            height_spin = self.wand_cam_table.cellWidget(row, 4)
            height = height_spin.value() if height_spin else 800
            
            camera_settings[cam_id] = {'focal': focal, 'width': width, 'height': height}
        
        # Use first camera's image size for the calibrator (or largest resolution)
        if camera_settings:
            first_cam = list(camera_settings.values())[0]
            h, w = first_cam['height'], first_cam['width']
            self.wand_calibrator.image_size = (h, w)
            init_focal = first_cam['focal']
            # Store all camera settings for per-camera focal length initialization
            self.wand_calibrator.camera_settings = camera_settings
            print(f"Using Camera Settings: image_size=({h},{w}), init_focal={init_focal}")
            print(f"  Per-camera focal lengths: {[(k, v['focal']) for k, v in camera_settings.items()]}")
        else:
            QMessageBox.warning(self, "No Cameras", "No camera settings found.")
            return

        wand_len = self.wand_len_spin.value()
        
        # Auto-filter: collect marked frames from error table and apply filter
        self._apply_current_ui_filter()
        
        # NOTE: Do NOT reset filter if frames_to_remove is empty. 
        # This allows cumulative filtering (i.e., previously removed frames stay removed).
        
        dist_coeff_num = self.dist_coeff_spin.value()
        print(f"Running Wand Calibration with length {wand_len}mm, f0={init_focal}px, dist_coeffs={dist_coeff_num}...")
        self.status_label.setText(f"Calibrating (L={wand_len}mm, f0={init_focal}px, k={dist_coeff_num})...")
    
        # Store precalibration state
        self._is_precalibrating = precalibrate
    
        # Disable button to prevent double click
        sender = self.sender()
        if sender: sender.setEnabled(False)
        self._calib_btn_sender = sender
        
        self.wand_calibrator.dist_coeff_num = dist_coeff_num
        
        # Use Worker
        from .wand_calibrator import CalibrationWorker
        self._calib_worker = CalibrationWorker(self.wand_calibrator, wand_len, init_focal, precalibrate=precalibrate)
        self._calib_worker.finished_signal.connect(self._on_calibration_finished)
        self._calib_worker.cost_signal.connect(self._on_cost_update)
        
        self._calib_worker.start()
        
        # Create custom progress dialog with cost plot
        self._create_calibration_dialog()
    
    def _save_filtered_points(self):
        """Save wand points to CSV. Includes all 'Raw' points and 'Filtered_Small/Large' for active frames."""
        if not hasattr(self, 'wand_calibrator'):
             QMessageBox.warning(self, "No Data", "No calibration data available.")
             return

        # Apply current UI filter marks before saving
        self._apply_current_ui_filter()
        
        raw_points = self.wand_calibrator.wand_points
        if not raw_points:
            QMessageBox.information(self, "Info", "No data points to save.")
            return

        # Determine which frames are kept (Raw) vs filtered
        if self.wand_calibrator.wand_points_filtered is not None:
            kept_frames = set(self.wand_calibrator.wand_points_filtered.keys())
        else:
            kept_frames = set(raw_points.keys()) # All kept

        # Prompt for save file (remembering last directory persistently)
        import os
        from PySide6.QtCore import QSettings
        settings = QSettings("OpenLPT", "GUI")
        init_dir = settings.value("last_wand_save_dir", "")
        print(f"[DEBUG] QSettings loaded last_wand_save_dir: '{init_dir}'")
        
        default_filename = "wandpoints_filtered.csv"
        if init_dir and isinstance(init_dir, str) and os.path.exists(init_dir):
             init_path = os.path.join(init_dir, default_filename)
             print(f"[DEBUG] Using init_path: '{init_path}'")
        else:
             init_path = default_filename
        
        filepath, _ = QFileDialog.getSaveFileName(self, "Save Points Data", init_path, "CSV Files (*.csv)")
        if not filepath:
            return
        
        # Remember directory persistently
        save_dir = os.path.dirname(filepath)
        save_dir = os.path.normpath(save_dir)
        settings.setValue("last_wand_save_dir", save_dir)
        settings.sync() # Force write
        print(f"[DEBUG] Saved last_wand_save_dir: '{save_dir}'")
        
        self.last_save_dir = save_dir
            
        # Collect data
        all_rows = []
        
        # Columns: Frame, Camera, Status, PointIdx, X, Y, Radius, Metric
        # Iterate over RAW points (superset)
        for fid in sorted(raw_points.keys()):
            is_kept = (fid in kept_frames)
            
            cams = raw_points[fid]
            for cam_idx in sorted(cams.keys()):
                pts_list = cams[cam_idx]
                
                # 1. Write RAW entries (ALL points)
                for p_idx, pt in enumerate(pts_list):
                    x, y = pt[0], pt[1]
                    r = pt[2] if len(pt) > 2 else 0
                    row = [fid, cam_idx, "Raw", p_idx, f"{x:.4f}", f"{y:.4f}", f"{r:.4f}", f"{0.0:.4f}"]
                    all_rows.append(row)
                
                # 2. Write FILTERED entries (Only kept frames)
                if is_kept and len(pts_list) >= 2:
                    # Sort by radius to identify Small/Large
                    # (Note: This assumes we are dealing with a standard 2-point wand)
                    pts_sorted = sorted(pts_list, key=lambda p: p[2] if len(p) > 2 else 0)
                    
                    # We expect exactly 2 points for a valid wand frame.
                    # If more, we might take the smallest and largest? 
                    # For now, let's take the first two (Smallest) and label them? 
                    # Or assume exactly 2.
                    # Loader logic checks len(pts) != 2 => invalid. 
                    # So we should probably output exactly 2 if we want it to work.
                    
                    # Let's take the smallest and the largest? Or just the first two?
                    # Screenshot implies Small and Large.
                    pt_small = pts_sorted[0]
                    pt_large = pts_sorted[-1] # Largest
                    
                    # Small
                    x, y, r = pt_small[0], pt_small[1], (pt_small[2] if len(pt_small)>2 else 0)
                    row_s = [fid, cam_idx, "Filtered_Small", 0, f"{x:.4f}", f"{y:.4f}", f"{r:.4f}", f"{0.0:.4f}"]
                    all_rows.append(row_s)
                    
                    # Large
                    x, y, r = pt_large[0], pt_large[1], (pt_large[2] if len(pt_large)>2 else 0)
                    row_l = [fid, cam_idx, "Filtered_Large", 1, f"{x:.4f}", f"{y:.4f}", f"{r:.4f}", f"{0.0:.4f}"]
                    all_rows.append(row_l)
                    
        # Write
        import csv
        header = ["Frame", "Camera", "Status", "PointIdx", "X", "Y", "Radius", "Metric"]
        
        try:
            with open(filepath, 'w', newline='') as f:
                writer = csv.writer(f)
                writer.writerow(header)
                writer.writerows(all_rows)
            
            QMessageBox.information(self, "Success", f"Saved {len(all_rows)} points to:\n{filepath}")
        except Exception as e:
             QMessageBox.critical(self, "Error", f"Failed to save file:\n{e}")

    def _create_calibration_dialog(self):
        """Create a dialog with matplotlib cost plot."""
        from PySide6.QtWidgets import QDialog, QVBoxLayout, QLabel
        from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg
        from matplotlib.figure import Figure
        
        self._calib_dialog = QDialog(self)
        self._calib_dialog.setWindowTitle("Calibrating...")
        self._calib_dialog.setModal(True)
        self._calib_dialog.setMinimumSize(400, 300)
        self._calib_dialog.setStyleSheet("background-color: #000000;")
        
        layout = QVBoxLayout(self._calib_dialog)
        layout.setContentsMargins(10, 10, 10, 10)
        
        # Matplotlib figure for cost curve
        self._cost_fig = Figure(figsize=(4, 2.5), facecolor='#000000')
        self._cost_canvas = FigureCanvasQTAgg(self._cost_fig)
        self._cost_ax = self._cost_fig.add_subplot(111)
        self._cost_ax.set_facecolor('#000000')
        self._cost_ax.set_xlabel('Iteration', color='white', fontsize=9)
        self._cost_ax.set_ylabel('Cost', color='white', fontsize=9)
        self._cost_ax.tick_params(colors='white', labelsize=8)
        for spine in self._cost_ax.spines.values():
            spine.set_color('#444')
        self._cost_ax.set_title('Cost vs Iteration', color='white', fontsize=10)
        
        # Initialize data and line
        self._cost_iterations = []
        self._cost_values = []
        self._cost_line, = self._cost_ax.plot([], [], 'c-', linewidth=1.5)
        
        self._cost_fig.tight_layout()
        layout.addWidget(self._cost_canvas)
        
        # Status label (below plot, centered)
        self._calib_status_label = QLabel("Running Bundle Adjustment... Please wait.")
        self._calib_status_label.setStyleSheet("font-size: 12px; color: #00d4ff; background: transparent;")
        self._calib_status_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        layout.addWidget(self._calib_status_label)
    
        # Stop Button
        self._stop_calib_btn = QPushButton("Stop")
        self._stop_calib_btn.setStyleSheet("background-color: #d32f2f; color: white; font-weight: bold; padding: 5px;")
        self._stop_calib_btn.clicked.connect(self._stop_calibration)
        self._stop_calib_btn.setEnabled(False)  # Disabled until Phase 3
        layout.addWidget(self._stop_calib_btn)
        
        self._calib_dialog.show()
    
    def _stop_calibration(self):
        """Request stop of running calibration."""
        if hasattr(self, '_calib_worker') and self._calib_worker.isRunning():
            self._calib_status_label.setText("Stopping calibration...")
            self._calib_worker.stop()
            # Disable button to indicate request sent
            self._stop_calib_btn.setEnabled(False)
    
    def _on_cost_update(self, phase, stage, rmse):
        """Called when optimizer reports new cost value."""
        # Ensure main event loop processes events (keeps UI responsive)
        from PySide6.QtWidgets import QApplication
        QApplication.processEvents()
        
        # Determine iteration (hacky counter since we removed iter from signal)
        if hasattr(self, '_cost_iter_count'):
             self._cost_iter_count += 1
        else:
             self._cost_iter_count = 1
             
        iteration = self._cost_iter_count
        
        if not hasattr(self, '_cost_iterations'):
            return
        
        self._cost_iterations.append(iteration)
        self._cost_values.append(rmse) # Plot RMSE instead of cost (more intuitive?) or keep cost? 
        # Actually user didn't ask to change plot, just text. 
        # But previous signal was (iteration, cost, rmse). Now (phase, stage, rmse).
        # We lost 'cost'. Let's plot RMSE then.
        
        # Update line data (efficient - no recreating)
        self._cost_line.set_data(self._cost_iterations, self._cost_values)
        self._cost_ax.relim()
        self._cost_ax.autoscale_view()
        
        # Update status label
        if hasattr(self, '_calib_status_label'):
             if getattr(self, '_is_precalibrating', False) or phase == "Pre-Calibration":
                 self._calib_status_label.setText(f"Proj error: {rmse:.4f} px.")
             else:
                 self._calib_status_label.setText(f"{phase}, {stage}, Proj error: {rmse:.4f} px.")
        
        # Enable/disable Stop button based on phase (only Phase 3 allows stopping)
        if hasattr(self, '_stop_calib_btn'):
            if "Phase 3" in phase or "Refinement" in phase:
                self._stop_calib_btn.setEnabled(True)
                self._stop_calib_btn.setStyleSheet("background-color: #d32f2f; color: white; font-weight: bold; padding: 5px;")
            else:
                self._stop_calib_btn.setEnabled(False)
                self._stop_calib_btn.setStyleSheet("background-color: #555; color: #888; font-weight: bold; padding: 5px;")
        
        # Redraw canvas
        self._cost_canvas.draw_idle()

    def _on_calibration_finished(self, success, msg, res=None):
        from PySide6.QtWidgets import QMessageBox, QFileDialog
        
        self._calib_dialog.close()
        if hasattr(self, '_calib_btn_sender') and self._calib_btn_sender:
            self._calib_btn_sender.setEnabled(True)
            
        print(f"Calibration Result: {success}, {msg}")
        
        if success:
            self.status_label.setText("Calibration Successful!")
            
            # Update 3D Visualization and Error Table FIRST so user can see results in background
            self._update_3d_viz()
            self._populate_error_table()
            
            # If Pre-calibration, SKIP save prompt
            if getattr(self, '_is_precalibrating', False):
                QMessageBox.information(self, "Check Complete", "Pre-calibration check complete.\nReview errors in the table.")
                return

            # Prompt to save with custom buttons
            msg_box = QMessageBox(self)
            msg_box.setWindowTitle("Calibration Result")
            msg_box.setText(f"Calibration Optimized Successfully!\n\n{msg}\n\nDo you want to save the camera parameters?")
            btn_save = msg_box.addButton("Save", QMessageBox.AcceptRole)
            btn_not_save = msg_box.addButton("Not Save", QMessageBox.RejectRole)
            msg_box.exec()
            
            if msg_box.clickedButton() == btn_save:
                # Prompt to save
                output_dir = QFileDialog.getExistingDirectory(self, "Select Output Directory for Camera Parameters")
                if output_dir:
                    from pathlib import Path
                    # Create camFile subfolder
                    cam_file_dir = Path(output_dir) / "camFile"
                    cam_file_dir.mkdir(parents=True, exist_ok=True)
                    
                    for cam_idx in self.wand_calibrator.final_params.keys():
                        path = cam_file_dir / f"cam{cam_idx}.txt"
                        self.wand_calibrator.export_to_file(cam_idx, str(path))
                    print(f"Parameters saved to {cam_file_dir}")
                    QMessageBox.information(self, "Saved", f"Camera parameters saved to:\n{cam_file_dir}")
        else:
            self.status_label.setText(f"Calibration Failed: {msg}")
            QMessageBox.critical(self, "Calibration Failed", msg)

    def _update_3d_viz(self):
        """Update the 3D visualization with calibrated camera positions and wand points."""
        import numpy as np
        
        if not hasattr(self, 'calib_3d_view'):
            return
            
        # Get camera params
        cameras = {}
        if hasattr(self.wand_calibrator, 'final_params'):
            cameras = self.wand_calibrator.final_params
        
        if not cameras: 
            return # Minimal silent return

        # Get points
        points_3d = None
        if hasattr(self.wand_calibrator, 'points_3d') and self.wand_calibrator.points_3d is not None:
            points_3d = self.wand_calibrator.points_3d
            pass
        else:
            pass
            
        # Plot
        self.calib_3d_view.plot_calibration(cameras, points_3d)
        
        # Switch to 3D Tab
        if hasattr(self, 'vis_tabs'):
            self.vis_tabs.setCurrentWidget(self.calib_3d_view)
    
    def _populate_error_table(self):
        """Populate error table with per-frame errors after calibration."""
        errors = self.wand_calibrator.calculate_per_frame_errors()
        if not errors:
            self.frozen_table.setRowCount(0)
            return

        # Identify filtered frames to persist checkbox state
        filtered_out_frames = set()
        if getattr(self.wand_calibrator, 'wand_points_filtered', None) is not None:
             all_frames = set(self.wand_calibrator.wand_points.keys())
             active_frames = set(self.wand_calibrator.wand_points_filtered.keys())
             filtered_out_frames = all_frames - active_frames
        
        # Determine camera list from all frames
        cam_ids_set = set()
        for frame_err in errors.values():
            cam_ids_set.update(frame_err['cam_errors'].keys())
        cam_ids = sorted(cam_ids_set)
        
        if not cam_ids:
            self.error_table.setRowCount(0)
            self.frozen_table.setRowCount(0)
            return
        
        # Setup table columns: Frame, Cam1, Cam2, ..., Len Err, Remove
        # IMPORTANT: Clear old contents first to avoid stale visual data
        self.error_table.clearContents()
        self.frozen_table.clearContents()
        self.error_table.setSortingEnabled(False)  # Always disable sorting while populating
        self.frozen_table.setSortingEnabled(False)
        
        # Right Table: Frame + cams + len_err
        col_count = 1 + len(cam_ids) + 1  
        self.error_table.setColumnCount(col_count)
        
        headers = ["Frame"] + [f"Cam {c+1}" for c in cam_ids] + ["Len Err (mm)"]
        self.error_table.setHorizontalHeaderLabels(headers)
        
        # Frozen Table: Just "Remove"
        self.frozen_table.setColumnCount(1)
        self.frozen_table.setHorizontalHeaderLabels(["Del"])
        
        # Populate rows - Only show frames with calculated errors (filtered frames)
        # Removed frames will not appear in the table at all
        frame_ids = sorted(errors.keys())
        self.error_table.setRowCount(len(frame_ids))
        self.frozen_table.setRowCount(len(frame_ids))
        self._error_frame_map = {}
        
        # Set Row Heights to match
        for row in range(len(frame_ids)):
            self.error_table.setRowHeight(row, 24)
            self.frozen_table.setRowHeight(row, 24)
        
        for row, fid in enumerate(frame_ids):
            self._error_frame_map[row] = fid
            err = errors[fid]
            
            # --- Frozen Table: Remove checkbox (Col 0) ---
            chk_widget = QWidget()
            chk_layout = QHBoxLayout(chk_widget)
            chk_layout.setContentsMargins(0, 0, 0, 0)
            chk_layout.setAlignment(Qt.AlignmentFlag.AlignCenter)
            chk = QCheckBox()
            chk.setProperty('frame_id', fid)
            chk_layout.addWidget(chk)
            self.frozen_table.setCellWidget(row, 0, chk_widget)
            
            # --- Right Table: Frame ID (Col 0) ---
            item = NumericTableWidgetItem(str(fid))
            item.setFlags(item.flags() & ~Qt.ItemFlag.ItemIsEditable)
            self.error_table.setItem(row, 0, item)
            
            # Per-camera errors (Col 1..N)
            for i, cam_id in enumerate(cam_ids):
                val = err['cam_errors'].get(cam_id, float('nan'))
                item = NumericTableWidgetItem(f"{val:.2f}")
                item.setFlags(item.flags() & ~Qt.ItemFlag.ItemIsEditable)
                if val > self.filter_proj_spin.value():
                    item.setBackground(Qt.GlobalColor.darkRed)
                self.error_table.setItem(row, i + 1, item)
            
            # Length error (Col N+1)
            len_err = err['len_error']
            item = NumericTableWidgetItem(f"{len_err:.3f}")
            item.setFlags(item.flags() & ~Qt.ItemFlag.ItemIsEditable)
            if len_err > self.filter_len_spin.value():
                item.setBackground(Qt.GlobalColor.darkRed)
            self.error_table.setItem(row, len(cam_ids) + 1, item)
        
        # Sizing
        self.error_table.resizeColumnsToContents()
        self.error_table.horizontalHeader().setMinimumSectionSize(60)
        for c in range(col_count):
             if self.error_table.columnWidth(c) < 60:
                 self.error_table.setColumnWidth(c, 60)
                 
        # Enable sorting - frozen_table will sync via _sync_frozen_table_sort
        self.error_table.setSortingEnabled(True)
        # Frozen table has no direct sorting (synced from error_table)
        
        # IMPORTANT: Trigger sync now in case a sort indicator is already active
        # (repopulating data doesn't trigger sortIndicatorChanged signal)
        header = self.error_table.horizontalHeader()
        self._sync_frozen_table_sort(header.sortIndicatorSection(), header.sortIndicatorOrder())
    
    def _sync_frozen_table_sort(self, logical_index, order):
        """Handle sort request by manually sorting and repopulating BOTH tables."""
        from PySide6.QtWidgets import QCheckBox, QWidget, QHBoxLayout
        from PySide6.QtCore import Qt
        
        if not hasattr(self, 'wand_calibrator') or not hasattr(self.wand_calibrator, 'per_frame_errors'):
            return
        
        errors = getattr(self.wand_calibrator, 'per_frame_errors', {})
        if not errors:
            return
        
        # Collect current checkbox states by frame_id BEFORE repopulating
        checkbox_states = {}
        for row in range(self.frozen_table.rowCount()):
            widget = self.frozen_table.cellWidget(row, 0)
            if widget:
                chk = widget.findChild(QCheckBox)
                if chk:
                    fid = chk.property('frame_id')
                    if fid is not None:
                        checkbox_states[int(fid)] = chk.isChecked()
        
        # Build sortable list based on sort column
        # Columns: 0=Frame, 1..N=Cam errors, N+1=Len Err
        cam_ids_set = set()
        for frame_err in errors.values():
            cam_ids_set.update(frame_err['cam_errors'].keys())
        cam_ids = sorted(cam_ids_set)
        
        sortable_data = []
        for fid, err in errors.items():
            if logical_index == 0:
                # Sort by Frame ID
                sort_key = fid
            elif logical_index <= len(cam_ids):
                # Sort by camera error
                cam_id = cam_ids[logical_index - 1]
                sort_key = err['cam_errors'].get(cam_id, float('nan'))
            else:
                # Sort by length error
                sort_key = err['len_error']
            sortable_data.append((fid, sort_key, err))
        
        # Sort with NaN handling
        from PySide6.QtCore import Qt as QtCore
        reverse = (order == QtCore.SortOrder.DescendingOrder)
        
        def safe_sort_key(item):
            val = item[1]
            if isinstance(val, float) and (val != val):  # NaN check
                return (1, 0)  # Push NaN to end
            return (0, val)
        
        sortable_data.sort(key=safe_sort_key, reverse=reverse)
        
        # Clear and repopulate BOTH tables
        self.error_table.setSortingEnabled(False)  # Disable to prevent recursive trigger
        self.error_table.clearContents()
        self.frozen_table.clearContents()
        
        num_rows = len(sortable_data)
        self.error_table.setRowCount(num_rows)
        self.frozen_table.setRowCount(num_rows)
        
        for row, (fid, _, err) in enumerate(sortable_data):
            self.error_table.setRowHeight(row, 24)
            self.frozen_table.setRowHeight(row, 24)
            
            # --- Frozen Table: Checkbox ---
            chk_widget = QWidget()
            chk_layout = QHBoxLayout(chk_widget)
            chk_layout.setContentsMargins(0, 0, 0, 0)
            chk_layout.setAlignment(Qt.AlignmentFlag.AlignCenter)
            chk = QCheckBox()
            chk.setProperty('frame_id', fid)
            if checkbox_states.get(fid, False):
                chk.setChecked(True)
            chk_layout.addWidget(chk)
            self.frozen_table.setCellWidget(row, 0, chk_widget)
            
            # --- Error Table: Frame ID (Col 0) ---
            item = NumericTableWidgetItem(str(fid))
            item.setFlags(item.flags() & ~Qt.ItemFlag.ItemIsEditable)
            self.error_table.setItem(row, 0, item)
            
            # --- Error Table: Camera errors (Col 1..N) ---
            for i, cam_id in enumerate(cam_ids):
                val = err['cam_errors'].get(cam_id, float('nan'))
                item = NumericTableWidgetItem(f"{val:.2f}")
                item.setFlags(item.flags() & ~Qt.ItemFlag.ItemIsEditable)
                if val > self.filter_proj_spin.value():
                    item.setBackground(Qt.GlobalColor.darkRed)
                self.error_table.setItem(row, i + 1, item)
            
            # --- Error Table: Length error (Col N+1) ---
            len_err = err['len_error']
            item = NumericTableWidgetItem(f"{len_err:.3f}")
            item.setFlags(item.flags() & ~Qt.ItemFlag.ItemIsEditable)
            if len_err > self.filter_len_spin.value():
                item.setBackground(Qt.GlobalColor.darkRed)
            self.error_table.setItem(row, len(cam_ids) + 1, item)
        
        # Re-enable sorting (but since we repopulated, the built-in sorting is now a no-op)
        self.error_table.setSortingEnabled(True)
    
    def _on_error_table_clicked(self, row, col):
        """Handle click on error table row - show frame images with overlay."""
        # Frame ID is now in column 0 of error_table
        item = self.error_table.item(row, 0)
        if item is None:
            return
        try:
            fid = int(item.text())
        except:
            return
        
        # Check if image paths are available
        has_images = bool(self.wand_images) and any(self.wand_images.values())
        if not has_images:
            # Show inline warning instead of popup
            if hasattr(self, 'error_warning_label'):
                self.error_warning_label.setText("⚠ Load image paths in Point Detection to preview.")
                self.error_warning_label.setVisible(True)
            return
        else:
            # Hide warning if images are available
            if hasattr(self, 'error_warning_label'):
                self.error_warning_label.setVisible(False)
        
        # Determine strict mapping for tab switching
        num_cams = len(self.wand_images)
        target_cam = None
        
        # Camera columns start at index 1 (0 is Frame ID)
        if 1 <= col < 1 + num_cams:
            target_cam = col - 1  # Column 1 = Camera 0
        
        # Display frame images with detection circles and reprojection overlay
        self._display_frame_with_overlay(fid, target_cam=target_cam)
    
    def _display_frame_with_overlay(self, frame_id, target_cam=None):
        """Display images for given frame with detection and reprojection overlay."""
        import cv2
        from PySide6.QtGui import QImage, QPixmap
        
        wand_data = self.wand_calibrator.wand_points_filtered or self.wand_calibrator.wand_points
        if frame_id not in wand_data:
            print(f"[Overlay] Frame {frame_id} not in wand_data (keys range: {min(wand_data.keys()) if wand_data else 0}-{max(wand_data.keys()) if wand_data else 0})")
            return
        
        obs = wand_data[frame_id]  # {cam_idx: [[x,y,r], [x,y,r]]}
        
        # Get 3D points for this frame
        frame_list = sorted(wand_data.keys())
        try:
            frame_i = frame_list.index(frame_id)
        except ValueError:
            print(f"[Overlay] Frame {frame_id} not found in sorted list")
            return
        
        pt3d_A = pt3d_B = None
        if hasattr(self.wand_calibrator, 'points_3d') and self.wand_calibrator.points_3d is not None:
            idx_A = frame_i * 2
            idx_B = frame_i * 2 + 1
            if idx_B < len(self.wand_calibrator.points_3d):
                pt3d_A = self.wand_calibrator.points_3d[idx_A]
                pt3d_B = self.wand_calibrator.points_3d[idx_B]
        
        for cam_idx, pts in obs.items():
            if cam_idx not in self.wand_images or not self.wand_images[cam_idx]:
                continue
            if frame_id >= len(self.wand_images[cam_idx]):
                continue
            
            img_path = self.wand_images[cam_idx][frame_id]
            img = cv2.imread(str(img_path))
            if img is None:
                continue
            
            # Draw detection circles (green)
            for pt in pts:
                x, y, r = int(pt[0]), int(pt[1]), int(pt[2]) if len(pt) > 2 else 20
                cv2.circle(img, (x, y), r, (0, 255, 0), 2)
                cv2.circle(img, (x, y), 3, (0, 255, 0), -1)
            
            # Draw reprojections (red cross) if calibration done
            if pt3d_A is not None and cam_idx in self.wand_calibrator.final_params:
                p = self.wand_calibrator.final_params[cam_idx]
                R, T, K, dist = p['R'], p['T'], p['K'], p['dist']
                rvec, _ = cv2.Rodrigues(R)
                
                for pt3d in [pt3d_A, pt3d_B]:
                    proj, _ = cv2.projectPoints(pt3d.reshape(1,3), rvec, T, K, dist)
                    px, py = int(proj[0,0,0]), int(proj[0,0,1])
                    cv2.drawMarker(img, (px, py), (0, 0, 255), cv2.MARKER_CROSS, 15, 2)
            
            # Convert to QPixmap and display
            img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            h, w, ch = img_rgb.shape
            qimg = QImage(img_rgb.data, w, h, ch * w, QImage.Format.Format_RGB888)
            pixmap = QPixmap.fromImage(qimg)
            
            # Scale to fit label
            if cam_idx in self.cam_vis_labels:
                lbl = self.cam_vis_labels[cam_idx]
                scaled = pixmap.scaled(lbl.size(), Qt.AspectRatioMode.KeepAspectRatio)
                lbl.setPixmap(scaled)
        
        # Switch to target camera tab (if specified) or first camera
        if hasattr(self, 'vis_tabs') and self.cam_vis_labels:
            switch_cam = target_cam if target_cam is not None and target_cam in self.cam_vis_labels else 0
            # Tab index matches camera index (0 = Cam 1, 1 = Cam 2, etc.)
            if switch_cam in self.cam_vis_labels:
                self.vis_tabs.setCurrentIndex(switch_cam)
    
    def _auto_update_filter_marks(self, *args):
        """Auto-update Remove checkboxes when filter criteria change."""
        if not hasattr(self.wand_calibrator, 'per_frame_errors') or not self.wand_calibrator.per_frame_errors:
            return  # No error data yet, skip silently
        
        proj_thresh = self.filter_proj_spin.value() if self.filter_proj_check.isChecked() else float('inf')
        len_thresh = self.filter_len_spin.value() if self.filter_len_check.isChecked() else float('inf')
        
        marked_count = 0
        # Update checkboxes in table
        for row in range(self.error_table.rowCount()):
            # Frame ID is in column 0 (not column 1!)
            item = self.error_table.item(row, 0)
            if item is None:
                continue
            try:
                fid = int(item.text())
            except:
                continue
            
            err = self.wand_calibrator.per_frame_errors.get(fid, {})
            max_proj = max(err.get('cam_errors', {}).values(), default=0)
            len_err = err.get('len_error', 0)
            
            should_remove = bool((max_proj > proj_thresh) or (len_err > len_thresh))
            if should_remove:
                marked_count += 1
            
            # Checkbox is in frozen_table column 0 (not error_table!)
            widget = self.frozen_table.cellWidget(row, 0)
            if widget:
                chk = widget.findChild(QCheckBox)
                if chk:
                    chk.setChecked(should_remove)
        
        total = self.error_table.rowCount()
        self.status_label.setText(f"Marked {marked_count}/{total} frames for removal.")
    
    def _save_filtered_data(self):
        """Save filtered data by removing marked frames."""
        frames_to_remove = set()
        for row in range(self.error_table.rowCount()):
            # Checkbox is in column 0
            widget = self.error_table.cellWidget(row, 0)
            if widget:
                chk = widget.findChild(QCheckBox)
                if chk and chk.isChecked():
                    # Get fid from table cell (column 1) - works after sorting
                    item = self.error_table.item(row, 1)
                    if item:
                        try:
                            fid = int(item.text())
                            frames_to_remove.add(fid)
                        except:
                            pass
        
        if not frames_to_remove:
            QMessageBox.information(self, "No Changes", "No frames marked for removal.")
            return
        
        remaining = self.wand_calibrator.apply_filter(frames_to_remove)
        QMessageBox.information(self, "Filtered", 
            f"Removed {len(frames_to_remove)} frames.\n{remaining} frames remaining.\n\nRun Calibration again to use filtered data.")
        self.status_label.setText(f"{remaining} frames ready. Re-run calibration.")

    def resizeEvent(self, event):
        super().resizeEvent(event)
        if hasattr(self, 'plate_vis_label') and self.tabs.currentIndex() == 0:
            if self.plate_img_list.currentRow() >= 0:
                self._display_plate_image(self.plate_img_list.currentRow())

    def create_wand_tab_v2_OLD_DUPLICATE_IGNORE(self):
        """Create the Wand Calibration tab (Multi-Camera) - Tabbed Interface."""
        tab = QWidget()
        layout = QHBoxLayout(tab)
        
        # 1. Visualization (LEFT)
        vis_frame = QFrame()
        vis_frame.setStyleSheet("background-color: #000000; border: 1px solid #333;")
        vis_layout = QVBoxLayout(vis_frame)
        vis_layout.setContentsMargins(0,0,0,0)

        self.vis_tabs = QTabWidget()
        self.vis_tabs.setStyleSheet("""
            QTabWidget::pane { border: 0; }
            QTabBar::tab { background: #222; color: #888; padding: 5px; }
            QTabBar::tab:selected { background: #444; color: #fff; }
        """)
        vis_layout.addWidget(self.vis_tabs)
        
        self.cam_vis_labels = {} # {cam_idx: QLabel}
        
        # 2. Controls Panel (RIGHT)
        right_panel = QWidget()
        right_panel.setFixedWidth(370)
        right_layout = QVBoxLayout(right_panel)
        right_layout.setContentsMargins(0, 0, 0, 0)
        
        controls_tabs = QTabWidget()
        controls_tabs.setStyleSheet("""
             QTabWidget::pane { border: 1px solid #444; background: #222; }
             QTabBar::tab { background: #333; color: #aaa; padding: 8px; min-width: 100px; }
             QTabBar::tab:selected { background: #444; color: #fff; border-bottom: 2px solid #00d4ff; }
        """)

        # --- Tab 1: Detection ---
        det_tab = QWidget()
        det_layout = QVBoxLayout(det_tab)
        det_layout.setSpacing(10)
        det_layout.setContentsMargins(10, 10, 10, 10)
        
        # Conf Group
        conf_group = QGroupBox("Detection Settings")
        conf_layout = QFormLayout(conf_group)
        
        self.wand_num_cams = QSpinBox()
        self._apply_input_style(self.wand_num_cams)
        self.wand_num_cams.setValue(4)
        self.wand_num_cams.setRange(1, 16)
        self.wand_num_cams.valueChanged.connect(self._update_wand_table)
        
        # Wand Type (Bright/Dark)
        self.wand_type_combo = QComboBox()
        self._apply_input_style(self.wand_type_combo)
        self.wand_type_combo.addItems(["Dark on Bright", "Bright on Dark"])
        
        # Circle Radius Range (Range Slider)
        self.radius_range = RangeSlider(min_val=1, max_val=500, initial_min=20, initial_max=200, suffix=" px")
        
        # Sensitivity slider
        from .widgets import SimpleSlider
        self.sensitivity_slider = SimpleSlider(min_val=0.5, max_val=1.0, initial=0.85, decimals=2)
        
        conf_layout.addRow("Num Cameras:", self.wand_num_cams)
        conf_layout.addRow("Wand Type:", self.wand_type_combo)
        conf_layout.addRow("Radius Range:", self.radius_range)
        conf_layout.addRow("Sensitivity:", self.sensitivity_slider)
        det_layout.addWidget(conf_group)
        
        # Table
        det_layout.addWidget(QLabel("Camera Images:"))
        self.wand_cam_table = QTableWidget()
        self.wand_cam_table.setColumnCount(3)
        self.wand_cam_table.setHorizontalHeaderLabels(["", "Camera", "Source"])
        header = self.wand_cam_table.horizontalHeader()
        header.setSectionResizeMode(0, QHeaderView.ResizeMode.ResizeToContents)
        header.setSectionResizeMode(1, QHeaderView.ResizeMode.Stretch)
        header.setSectionResizeMode(2, QHeaderView.ResizeMode.ResizeToContents)
        self.wand_cam_table.verticalHeader().setVisible(False)
        self.wand_cam_table.setSelectionMode(QTableWidget.SelectionMode.NoSelection)
        self.wand_cam_table.setShowGrid(False)
        self.wand_cam_table.setStyleSheet("background-color: transparent; border: 1px solid #333;")
        self.wand_cam_table.setFixedHeight(100) 
        self._update_wand_table(4)
        det_layout.addWidget(self.wand_cam_table)
        
        # Frame List
        det_layout.addWidget(QLabel("Frame List (Click to Preview):"))
        self.frame_table = QTableWidget()
        self.frame_table.setColumnCount(2)
        self.frame_table.setHorizontalHeaderLabels(["Index", "Filename"])
        self.frame_table.horizontalHeader().setSectionResizeMode(1, QHeaderView.ResizeMode.Stretch)
        self.frame_table.verticalHeader().setVisible(False)
        self.frame_table.setSelectionBehavior(QTableWidget.SelectionBehavior.SelectRows)
        self.frame_table.setSelectionMode(QTableWidget.SelectionMode.SingleSelection)
        self.frame_table.setStyleSheet("background-color: transparent; border: 1px solid #333;")
        self.frame_table.cellClicked.connect(self._on_frame_table_clicked)
        self.frame_table.setFixedHeight(120) 
        det_layout.addWidget(self.frame_table)

        # Actions
        self.btn_detect_single = QPushButton("Test Detect (Current Frame)")
        self.btn_detect_single.setStyleSheet("background-color: #2a3f5f;")
        self.btn_detect_single.clicked.connect(self._detect_single_frame)
        det_layout.addWidget(self.btn_detect_single)

        self.btn_process_wand = QPushButton("1. Process All Frames / Resume")
        self.btn_process_wand.setStyleSheet("background-color: #2a3f5f; font-weight: bold;")
        self.btn_process_wand.clicked.connect(self._process_wand_frames)
        det_layout.addWidget(self.btn_process_wand)
        
        det_layout.addStretch()

        # --- Tab 2: Calibration ---
        cal_tab = QWidget()
        cal_layout = QVBoxLayout(cal_tab)
        cal_layout.setSpacing(15)
        cal_layout.setContentsMargins(10, 10, 10, 10)
        
        cal_group = QGroupBox("Calibration Settings")
        cal_form = QFormLayout(cal_group)
        
        self.wand_model_combo = QComboBox()
        self._apply_input_style(self.wand_model_combo)
        self.wand_model_combo.addItems(["Pinhole", "Polynomial"])
        
        self.wand_len_spin = QDoubleSpinBox()
        self._apply_input_style(self.wand_len_spin)
        self.wand_len_spin.setValue(500.0)
        self.wand_len_spin.setRange(10, 5000)
        self.wand_len_spin.setSuffix(" mm")
        
        # Image Resolution Manual Override
        self.img_width_spin = QSpinBox()
        self._apply_input_style(self.img_width_spin)
        self.img_width_spin.setRange(0, 10000)
        self.img_width_spin.setValue(0)
        self.img_width_spin.setSuffix(" px")
        self.img_width_spin.setToolTip("Set to 0 to auto-detect from loaded images.")
        
        self.img_height_spin = QSpinBox()
        self._apply_input_style(self.img_height_spin)
        self.img_height_spin.setRange(0, 10000)
        self.img_height_spin.setValue(0)
        self.img_height_spin.setSuffix(" px")
        self.img_height_spin.setToolTip("Set to 0 to auto-detect from loaded images.")

        cal_form.addRow("Camera Model:", self.wand_model_combo)
        cal_form.addRow("Wand Length:", self.wand_len_spin)
        cal_form.addRow("Image Width:", self.img_width_spin)
        cal_form.addRow("Image Height:", self.img_height_spin)
        cal_layout.addWidget(cal_group)
        
        cal_layout.addStretch()
        
        # Load Points Button
        self.btn_load_points = QPushButton("Load Wand Points (from CSV)")
        self.btn_load_points.setStyleSheet("background-color: #4a4a4a; border: 1px solid #666;")
        self.btn_load_points.clicked.connect(self._load_wand_points_for_calibration)
        cal_layout.addWidget(self.btn_load_points)
        
        # Precalibrate Check
        self.btn_precalibrate = QPushButton("Precalibrate to Check")
        self.btn_precalibrate.setStyleSheet("background-color: #ff9800; color: #000000; font-weight: bold; padding: 10px;")
        self.btn_precalibrate.clicked.connect(lambda: self._run_wand_calibration(precalibrate=True))
        cal_layout.addWidget(self.btn_precalibrate)

        # Run Calibration
        self.btn_calibrate_wand = QPushButton("2. Run Calibration")
        self.btn_calibrate_wand.setStyleSheet("background-color: #00d4ff; color: #000000; font-weight: bold; height: 50px; font-size: 14px;")
        self.btn_calibrate_wand.clicked.connect(lambda: self._run_wand_calibration(precalibrate=False))
        cal_layout.addWidget(self.btn_calibrate_wand)
        
        cal_layout.addStretch()

        # Add tabs
        controls_tabs.addTab(det_tab, "Point Detection")
        controls_tabs.addTab(cal_tab, "Calibration")
        
        right_layout.addWidget(controls_tabs)
        
        # Progress Label
        self.status_label = QLabel("Status: Idle")
        self.status_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.status_label.setStyleSheet("color: #888; font-size: 11px; margin-top: 5px;")
        right_layout.addWidget(self.status_label)

        layout.addWidget(vis_frame, stretch=2)
        layout.addWidget(right_panel)
        
        return tab

    def _load_wand_points_for_calibration_OLD_DUPLICATE(self):
        """Prompt to load a CSV file, populate wand points, and ready for calibration."""
        file_path, _ = QFileDialog.getOpenFileName(self, "Load Wand Points", "", "CSV Files (*.csv)")
        if not file_path:
            return
            
        success, msg = self.wand_calibrator.load_wand_data_from_csv(file_path)
        if success:
            # Ensure calibrator has access to cameras/size
            if self.wand_images:
                 self.wand_calibrator.cameras = {}
                 for c, imgs in self.wand_images.items():
                     self.wand_calibrator.cameras[c] = {'images': imgs}
                     
            QMessageBox.information(self, "Success", msg + "\nYou can now run calibration.")
            count_frames = len(self.wand_calibrator.wand_points)
            self.status_label.setText(f"Loaded {count_frames} frames. Ready to calibrate.")
            # Switch to Vis tab 0 if needed or show loaded points on current frame?
            # Ideally we'd visualize, but just loading is fine for now.
        else:
            QMessageBox.critical(self, "Error", f"Failed to load points:\n{msg}")
