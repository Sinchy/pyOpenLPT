"""
Tracking Settings View
"""

from PySide6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QLabel, QFrame,
    QPushButton, QGroupBox, QGridLayout, QSpinBox, QDoubleSpinBox,
    QCheckBox, QScrollArea, QFileDialog, QLineEdit, QTabWidget
)
from PySide6.QtCore import Qt


class TrackingSettingsView(QWidget):
    """View for configuring tracking parameters."""
    
    def __init__(self):
        super().__init__()
        self._setup_ui()
    
    def _setup_ui(self):
        layout = QHBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(16)
        
        # === Main Settings Area (Scrollable) ===
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setFrameShape(QFrame.Shape.NoFrame)
        
        scroll_content = QWidget()
        scroll_layout = QVBoxLayout(scroll_content)
        scroll_layout.setSpacing(16)
        
        # Title
        title = QLabel("Tracking Settings")
        title.setObjectName("sectionTitle")
        scroll_layout.addWidget(title)
        
        # Config file
        config_group = QGroupBox("Configuration File")
        config_layout = QHBoxLayout(config_group)
        self.config_path = QLineEdit()
        self.config_path.setPlaceholderText("Select configuration file...")
        config_layout.addWidget(self.config_path)
        browse_btn = QPushButton("📂")
        browse_btn.setFixedWidth(40)
        browse_btn.clicked.connect(self._browse_config)
        config_layout.addWidget(browse_btn)
        load_btn = QPushButton("Load")
        load_btn.clicked.connect(self._load_config)
        config_layout.addWidget(load_btn)
        scroll_layout.addWidget(config_group)
        
        # Create tabs for different parameter groups
        tabs = QTabWidget()
        
        # === Basic Tab ===
        basic_widget = QWidget()
        basic_layout = QVBoxLayout(basic_widget)
        
        basic_group = QGroupBox("Basic Settings")
        basic_grid = QGridLayout(basic_group)
        
        basic_grid.addWidget(QLabel("Number of Cameras:"), 0, 0)
        self.n_cam_spin = QSpinBox()
        self.n_cam_spin.setRange(2, 16)
        self.n_cam_spin.setValue(4)
        basic_grid.addWidget(self.n_cam_spin, 0, 1)
        
        basic_grid.addWidget(QLabel("Frame Start:"), 1, 0)
        self.frame_start_spin = QSpinBox()
        self.frame_start_spin.setRange(0, 100000)
        basic_grid.addWidget(self.frame_start_spin, 1, 1)
        
        basic_grid.addWidget(QLabel("Frame End:"), 2, 0)
        self.frame_end_spin = QSpinBox()
        self.frame_end_spin.setRange(1, 100000)
        self.frame_end_spin.setValue(1000)
        basic_grid.addWidget(self.frame_end_spin, 2, 1)
        
        basic_grid.addWidget(QLabel("FPS:"), 3, 0)
        self.fps_spin = QSpinBox()
        self.fps_spin.setRange(1, 10000)
        self.fps_spin.setValue(1000)
        basic_grid.addWidget(self.fps_spin, 3, 1)
        
        basic_grid.addWidget(QLabel("Voxel to mm:"), 4, 0)
        self.voxel_spin = QDoubleSpinBox()
        self.voxel_spin.setDecimals(6)
        self.voxel_spin.setRange(0.000001, 100)
        self.voxel_spin.setValue(0.001)
        basic_grid.addWidget(self.voxel_spin, 4, 1)
        
        basic_layout.addWidget(basic_group)
        basic_layout.addStretch()
        tabs.addTab(basic_widget, "Basic")
        
        # === IPR Tab ===
        ipr_widget = QWidget()
        ipr_layout = QVBoxLayout(ipr_widget)
        
        ipr_group = QGroupBox("IPR Parameters")
        ipr_grid = QGridLayout(ipr_group)
        
        ipr_grid.addWidget(QLabel("Cameras to Reduce:"), 0, 0)
        self.ipr_reduce_spin = QSpinBox()
        self.ipr_reduce_spin.setRange(0, 4)
        self.ipr_reduce_spin.setValue(1)
        ipr_grid.addWidget(self.ipr_reduce_spin, 0, 1)
        
        ipr_grid.addWidget(QLabel("IPR Loops:"), 1, 0)
        self.ipr_loop_spin = QSpinBox()
        self.ipr_loop_spin.setRange(1, 20)
        self.ipr_loop_spin.setValue(4)
        ipr_grid.addWidget(self.ipr_loop_spin, 1, 1)
        
        ipr_grid.addWidget(QLabel("Reduced Loops:"), 2, 0)
        self.ipr_reduced_spin = QSpinBox()
        self.ipr_reduced_spin.setRange(1, 20)
        self.ipr_reduced_spin.setValue(2)
        ipr_grid.addWidget(self.ipr_reduced_spin, 2, 1)
        
        ipr_layout.addWidget(ipr_group)
        ipr_layout.addStretch()
        tabs.addTab(ipr_widget, "IPR")
        
        # === STB Tab ===
        stb_widget = QWidget()
        stb_layout = QVBoxLayout(stb_widget)
        
        stb_group = QGroupBox("STB Parameters")
        stb_grid = QGridLayout(stb_group)
        
        stb_grid.addWidget(QLabel("Search Radius (obj):"), 0, 0)
        self.stb_radius_obj = QDoubleSpinBox()
        self.stb_radius_obj.setDecimals(4)
        self.stb_radius_obj.setRange(0.0001, 100)
        self.stb_radius_obj.setValue(0.1)
        stb_grid.addWidget(self.stb_radius_obj, 0, 1)
        
        stb_grid.addWidget(QLabel("Initial Frames:"), 1, 0)
        self.stb_initial_spin = QSpinBox()
        self.stb_initial_spin.setRange(4, 100)
        self.stb_initial_spin.setValue(4)
        stb_grid.addWidget(self.stb_initial_spin, 1, 1)
        
        stb_grid.addWidget(QLabel("Search Radius (track):"), 2, 0)
        self.stb_radius_track = QDoubleSpinBox()
        self.stb_radius_track.setDecimals(4)
        self.stb_radius_track.setRange(0.0001, 100)
        self.stb_radius_track.setValue(0.05)
        stb_grid.addWidget(self.stb_radius_track, 2, 1)
        
        stb_layout.addWidget(stb_group)
        stb_layout.addStretch()
        tabs.addTab(stb_widget, "STB")
        
        # === Shake Tab ===
        shake_widget = QWidget()
        shake_layout = QVBoxLayout(shake_widget)
        
        shake_group = QGroupBox("Shake Parameters")
        shake_grid = QGridLayout(shake_group)
        
        shake_grid.addWidget(QLabel("Shake Width:"), 0, 0)
        self.shake_width = QDoubleSpinBox()
        self.shake_width.setDecimals(4)
        self.shake_width.setRange(0.0001, 10)
        self.shake_width.setValue(0.01)
        shake_grid.addWidget(self.shake_width, 0, 1)
        
        shake_grid.addWidget(QLabel("Shake Loops:"), 1, 0)
        self.shake_loops = QSpinBox()
        self.shake_loops.setRange(1, 20)
        self.shake_loops.setValue(4)
        shake_grid.addWidget(self.shake_loops, 1, 1)
        
        shake_grid.addWidget(QLabel("Ghost Threshold:"), 2, 0)
        self.shake_ghost = QDoubleSpinBox()
        self.shake_ghost.setDecimals(3)
        self.shake_ghost.setRange(0.001, 1.0)
        self.shake_ghost.setValue(0.1)
        shake_grid.addWidget(self.shake_ghost, 2, 1)
        
        shake_layout.addWidget(shake_group)
        shake_layout.addStretch()
        tabs.addTab(shake_widget, "Shake")
        
        scroll_layout.addWidget(tabs)
        scroll_layout.addStretch()
        
        scroll.setWidget(scroll_content)
        layout.addWidget(scroll, stretch=2)
        
        # === Right: Actions Panel ===
        actions_frame = QFrame()
        actions_frame.setObjectName("paramPanel")
        actions_frame.setFixedWidth(280)
        actions_layout = QVBoxLayout(actions_frame)
        actions_layout.setSpacing(12)
        
        actions_title = QLabel("Actions")
        actions_title.setObjectName("sectionTitle")
        actions_layout.addWidget(actions_title)
        
        actions_layout.addWidget(QLabel("Output Path:"))
        self.output_path = QLineEdit()
        self.output_path.setPlaceholderText("Select output directory...")
        actions_layout.addWidget(self.output_path)
        
        import qtawesome as qta
        output_browse = QPushButton(" Browse")
        output_browse.setIcon(qta.icon("fa5s.folder-open", color="white"))
        output_browse.clicked.connect(self._browse_output)
        actions_layout.addWidget(output_browse)
        
        actions_layout.addStretch()
        
        validate_btn = QPushButton(" Validate Settings")
        validate_btn.setIcon(qta.icon("fa5s.check", color="white"))
        actions_layout.addWidget(validate_btn)
        
        save_btn = QPushButton(" Save Configuration")
        save_btn.setIcon(qta.icon("fa5s.save", color="white"))
        save_btn.setObjectName("primaryButton")
        actions_layout.addWidget(save_btn)
        
        layout.addWidget(actions_frame)
    
    def _browse_config(self):
        file_path, _ = QFileDialog.getOpenFileName(
            self, "Select Configuration File", "", "Config Files (*.cfg);;All Files (*)"
        )
        if file_path:
            self.config_path.setText(file_path)
    
    def _load_config(self):
        # TODO: Implement config loading
        pass
    
    def _browse_output(self):
        dir_path = QFileDialog.getExistingDirectory(self, "Select Output Directory")
        if dir_path:
            self.output_path.setText(dir_path)
