"""
Tracking View - Run and monitor tracking process
"""

from PySide6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QLabel, QFrame,
    QPushButton, QGroupBox, QGridLayout, QProgressBar,
    QTextEdit, QSplitter
)
from PySide6.QtCore import Qt


class TrackingView(QWidget):
    """View for running and monitoring the tracking process."""
    
    def __init__(self):
        super().__init__()
        self._setup_ui()
    
    def _setup_ui(self):
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(16)
        
        title = QLabel("Tracking")
        title.setObjectName("sectionTitle")
        layout.addWidget(title)
        
        splitter = QSplitter(Qt.Orientation.Horizontal)
        
        # Left: Camera Views Grid
        cameras_frame = QFrame()
        cameras_frame.setObjectName("viewFrame")
        cameras_layout = QVBoxLayout(cameras_frame)
        
        camera_grid = QGridLayout()
        for i in range(4):
            cam_frame = QFrame()
            cam_frame.setStyleSheet("""
                background-color: #0d1117;
                border: 1px solid #2d3a4a;
                border-radius: 6px;
                min-height: 200px;
            """)
            cam_layout = QVBoxLayout(cam_frame)
            cam_label = QLabel(f"Camera {i+1}")
            cam_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
            cam_label.setStyleSheet("color: #4a5a6a;")
            cam_layout.addWidget(cam_label)
            camera_grid.addWidget(cam_frame, i // 2, i % 2)
        
        cameras_layout.addLayout(camera_grid)
        splitter.addWidget(cameras_frame)
        
        # Right: Control Panel
        control_frame = QFrame()
        control_frame.setObjectName("paramPanel")
        control_frame.setFixedWidth(320)
        control_layout = QVBoxLayout(control_frame)
        
        # Progress
        progress_group = QGroupBox("Progress")
        progress_layout = QVBoxLayout(progress_group)
        self.overall_progress = QProgressBar()
        progress_layout.addWidget(self.overall_progress)
        self.frame_label = QLabel("Frame: 0 / 0")
        progress_layout.addWidget(self.frame_label)
        control_layout.addWidget(progress_group)
        
        # Controls
        import qtawesome as qta
        self.start_btn = QPushButton(" Start Tracking")
        self.start_btn.setIcon(qta.icon("fa5s.play", color="white"))
        self.start_btn.setObjectName("primaryButton")
        control_layout.addWidget(self.start_btn)
        
        btn_layout = QHBoxLayout()
        self.pause_btn = QPushButton(" Pause")
        self.pause_btn.setIcon(qta.icon("fa5s.pause", color="black")) # Pause often on yellow background
        self.stop_btn = QPushButton(" Stop")
        self.stop_btn.setIcon(qta.icon("fa5s.stop", color="white"))
        self.stop_btn.setObjectName("dangerButton")
        btn_layout.addWidget(self.pause_btn)
        btn_layout.addWidget(self.stop_btn)
        control_layout.addLayout(btn_layout)
        
        # Log
        log_group = QGroupBox("Log")
        log_layout = QVBoxLayout(log_group)
        self.log_text = QTextEdit()
        self.log_text.setReadOnly(True)
        self.log_text.setStyleSheet("background-color: #0d1117; font-family: Consolas;")
        log_layout.addWidget(self.log_text)
        control_layout.addWidget(log_group)
        
        splitter.addWidget(control_frame)
        splitter.setSizes([700, 320])
        layout.addWidget(splitter)
