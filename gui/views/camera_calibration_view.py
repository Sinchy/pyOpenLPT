"""
Camera Calibration View
"""

from PySide6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QLabel, QFrame,
    QPushButton, QGroupBox, QGridLayout, QLineEdit, QSpinBox,
    QDoubleSpinBox, QFileDialog, QListWidget, QSplitter
)
from PySide6.QtCore import Qt


class CameraCalibrationView(QWidget):
    """View for camera calibration functionality."""
    
    def __init__(self):
        super().__init__()
        self._setup_ui()
    
    def _setup_ui(self):
        layout = QHBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(16)
        
        # === Left: Camera Preview ===
        preview_frame = QFrame()
        preview_frame.setObjectName("viewFrame")
        preview_layout = QVBoxLayout(preview_frame)
        
        # Preview area
        self.preview_label = QLabel("Camera Preview")
        self.preview_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.preview_label.setStyleSheet("""
            background-color: #0d1117;
            color: #4a5a6a;
            font-size: 18px;
            min-height: 400px;
            border-radius: 8px;
        """)
        preview_layout.addWidget(self.preview_label, stretch=1)
        
        # Camera list
        camera_group = QGroupBox("Cameras")
        camera_layout = QVBoxLayout(camera_group)
        self.camera_list = QListWidget()
        self.camera_list.addItems(["Camera 1", "Camera 2", "Camera 3", "Camera 4"])
        camera_layout.addWidget(self.camera_list)
        preview_layout.addWidget(camera_group)
        
        layout.addWidget(preview_frame, stretch=2)
        
        # === Right: Parameters Panel ===
        params_frame = QFrame()
        params_frame.setObjectName("paramPanel")
        params_frame.setFixedWidth(320)
        params_layout = QVBoxLayout(params_frame)
        params_layout.setSpacing(16)
        
        # Title
        title = QLabel("Camera Calibration")
        title.setObjectName("sectionTitle")
        params_layout.addWidget(title)
        
        # Load camera parameters
        load_group = QGroupBox("Load Parameters")
        load_layout = QVBoxLayout(load_group)
        
        self.cam_path_edit = QLineEdit()
        self.cam_path_edit.setPlaceholderText("Select camera parameter file...")
        load_layout.addWidget(self.cam_path_edit)
        
        browse_btn = QPushButton("📂 Browse")
        browse_btn.clicked.connect(self._browse_camera_file)
        load_layout.addWidget(browse_btn)
        
        params_layout.addWidget(load_group)
        
        # VSC Calibration
        vsc_group = QGroupBox("Volume Self-Calibration (VSC)")
        vsc_layout = QGridLayout(vsc_group)
        
        vsc_layout.addWidget(QLabel("Min Track Length:"), 0, 0)
        self.min_track_spin = QSpinBox()
        self.min_track_spin.setRange(1, 1000)
        self.min_track_spin.setValue(30)
        vsc_layout.addWidget(self.min_track_spin, 0, 1)
        
        vsc_layout.addWidget(QLabel("Isolation Radius:"), 1, 0)
        self.isolation_spin = QDoubleSpinBox()
        self.isolation_spin.setRange(0.1, 100.0)
        self.isolation_spin.setValue(3.0)
        vsc_layout.addWidget(self.isolation_spin, 1, 1)
        
        vsc_layout.addWidget(QLabel("Max Reproj Error:"), 2, 0)
        self.reproj_spin = QDoubleSpinBox()
        self.reproj_spin.setRange(0.01, 10.0)
        self.reproj_spin.setValue(1.0)
        vsc_layout.addWidget(self.reproj_spin, 2, 1)
        
        params_layout.addWidget(vsc_group)
        
        # Action buttons
        params_layout.addStretch()
        
        import qtawesome as qta
        run_vsc_btn = QPushButton(" Run VSC Calibration")
        run_vsc_btn.setIcon(qta.icon("fa5s.play", color="white"))
        run_vsc_btn.setObjectName("primaryButton")
        params_layout.addWidget(run_vsc_btn)
        
        save_btn = QPushButton(" Save Calibration")
        save_btn.setIcon(qta.icon("fa5s.save", color="white"))
        params_layout.addWidget(save_btn)
        
        layout.addWidget(params_frame)
    
    def _browse_camera_file(self):
        """Open file dialog to select camera parameter file."""
        file_path, _ = QFileDialog.getOpenFileName(
            self,
            "Select Camera Parameter File",
            "",
            "All Files (*);;Text Files (*.txt);;Config Files (*.cfg)"
        )
        if file_path:
            self.cam_path_edit.setText(file_path)
