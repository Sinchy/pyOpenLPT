"""
Results View - 3D trajectory visualization and export
"""

from PySide6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QLabel, QFrame,
    QPushButton, QGroupBox, QComboBox, QSlider, QCheckBox
)
from PySide6.QtCore import Qt


class ResultsView(QWidget):
    """View for displaying results and 3D trajectory visualization."""
    
    def __init__(self):
        super().__init__()
        self._setup_ui()
    
    def _setup_ui(self):
        layout = QHBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(16)
        
        # Left: 3D Visualization Area
        viz_frame = QFrame()
        viz_frame.setObjectName("viewFrame")
        viz_layout = QVBoxLayout(viz_frame)
        
        # Placeholder for 3D view (will be replaced with OpenGL/VTK widget)
        self.viz_label = QLabel("3D Trajectory Visualization")
        self.viz_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.viz_label.setStyleSheet("""
            background-color: #0d1117;
            color: #00d4ff;
            font-size: 24px;
            min-height: 500px;
            border-radius: 8px;
        """)
        viz_layout.addWidget(self.viz_label)
        
        # View controls
        view_controls = QHBoxLayout()
        view_controls.addWidget(QPushButton("🔄 Reset View"))
        view_controls.addWidget(QPushButton("📸 Screenshot"))
        view_controls.addStretch()
        viz_layout.addLayout(view_controls)
        
        layout.addWidget(viz_frame, stretch=2)
        
        # Right: Settings Panel
        settings_frame = QFrame()
        settings_frame.setObjectName("paramPanel")
        settings_frame.setFixedWidth(300)
        settings_layout = QVBoxLayout(settings_frame)
        
        title = QLabel("Visualization Settings")
        title.setObjectName("sectionTitle")
        settings_layout.addWidget(title)
        
        # Color mapping
        color_group = QGroupBox("Color Mapping")
        color_layout = QVBoxLayout(color_group)
        self.color_combo = QComboBox()
        self.color_combo.addItems(["Velocity", "Time", "Track ID", "Solid Color"])
        color_layout.addWidget(self.color_combo)
        settings_layout.addWidget(color_group)
        
        # Display options
        display_group = QGroupBox("Display Options")
        display_layout = QVBoxLayout(display_group)
        display_layout.addWidget(QCheckBox("Show Particles"))
        display_layout.addWidget(QCheckBox("Show Trajectories"))
        display_layout.addWidget(QCheckBox("Show Axes"))
        display_layout.addWidget(QCheckBox("Show Bounding Box"))
        settings_layout.addWidget(display_group)
        
        # Track filter
        filter_group = QGroupBox("Track Filter")
        filter_layout = QVBoxLayout(filter_group)
        filter_layout.addWidget(QLabel("Min Length:"))
        self.min_len_slider = QSlider(Qt.Orientation.Horizontal)
        self.min_len_slider.setRange(1, 100)
        filter_layout.addWidget(self.min_len_slider)
        settings_layout.addWidget(filter_group)
        
        settings_layout.addStretch()
        
        # Export
        import qtawesome as qta
        export_group = QGroupBox("Export")
        export_layout = QVBoxLayout(export_group)
        
        btn_csv = QPushButton(" Export CSV")
        btn_csv.setIcon(qta.icon("fa5s.file-csv", color="white"))
        export_layout.addWidget(btn_csv)
        
        btn_anim = QPushButton(" Export Animation")
        btn_anim.setIcon(qta.icon("fa5s.video", color="white"))
        export_layout.addWidget(btn_anim)
        
        btn_vtk = QPushButton(" Export VTK")
        btn_vtk.setIcon(qta.icon("fa5s.cube", color="white"))
        export_layout.addWidget(btn_vtk)
        
        settings_layout.addWidget(export_group)
        
        layout.addWidget(settings_frame)
