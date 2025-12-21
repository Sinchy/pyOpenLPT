"""
Image Preprocessing View
"""

from PySide6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QLabel, QFrame,
    QPushButton, QGroupBox, QGridLayout, QSlider, QSpinBox,
    QCheckBox, QComboBox, QFileDialog
)
from PySide6.QtCore import Qt


class ImagePreprocessingView(QWidget):
    """View for image preprocessing functionality."""
    
    def __init__(self):
        super().__init__()
        self._setup_ui()
    
    def _setup_ui(self):
        layout = QHBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(16)
        
        # === Left: Image Preview ===
        preview_frame = QFrame()
        preview_frame.setObjectName("viewFrame")
        preview_layout = QVBoxLayout(preview_frame)
        
        # Before/After toggle
        toggle_layout = QHBoxLayout()
        self.before_btn = QPushButton("Original")
        self.before_btn.setCheckable(True)
        self.before_btn.setChecked(True)
        self.after_btn = QPushButton("Processed")
        self.after_btn.setCheckable(True)
        toggle_layout.addWidget(self.before_btn)
        toggle_layout.addWidget(self.after_btn)
        toggle_layout.addStretch()
        preview_layout.addLayout(toggle_layout)
        
        # Image display area
        self.image_label = QLabel("Load an image to preview")
        self.image_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.image_label.setStyleSheet("""
            background-color: #0d1117;
            color: #4a5a6a;
            font-size: 18px;
            min-height: 500px;
            border-radius: 8px;
        """)
        preview_layout.addWidget(self.image_label, stretch=1)
        
        layout.addWidget(preview_frame, stretch=2)
        
        # === Right: Parameters Panel ===
        params_frame = QFrame()
        params_frame.setObjectName("paramPanel")
        params_frame.setFixedWidth(320)
        params_layout = QVBoxLayout(params_frame)
        params_layout.setSpacing(16)
        
        # Title
        title = QLabel("Image Preprocessing")
        title.setObjectName("sectionTitle")
        params_layout.addWidget(title)
        
        # Image source
        source_group = QGroupBox("Image Source")
        source_layout = QVBoxLayout(source_group)
        
        import qtawesome as qta
        browse_btn = QPushButton(" Load Images")
        browse_btn.setIcon(qta.icon("fa5s.folder-open", color="white"))
        browse_btn.clicked.connect(self._browse_images)
        source_layout.addWidget(browse_btn)
        
        self.image_count_label = QLabel("0 images loaded")
        self.image_count_label.setStyleSheet("color: #a0a0a0;")
        source_layout.addWidget(self.image_count_label)
        
        params_layout.addWidget(source_group)
        
        # Background subtraction
        bg_group = QGroupBox("Background Subtraction")
        bg_layout = QVBoxLayout(bg_group)
        
        self.bg_enabled = QCheckBox("Enable")
        self.bg_enabled.setChecked(True)
        bg_layout.addWidget(self.bg_enabled)
        
        bg_method_layout = QHBoxLayout()
        bg_method_layout.addWidget(QLabel("Method:"))
        self.bg_method = QComboBox()
        self.bg_method.addItems(["Average", "Median", "Min", "Rolling"])
        bg_method_layout.addWidget(self.bg_method)
        bg_layout.addLayout(bg_method_layout)
        
        params_layout.addWidget(bg_group)
        
        # Particle detection
        detect_group = QGroupBox("Particle Detection")
        detect_layout = QGridLayout(detect_group)
        
        detect_layout.addWidget(QLabel("Threshold:"), 0, 0)
        self.threshold_slider = QSlider(Qt.Orientation.Horizontal)
        self.threshold_slider.setRange(0, 255)
        self.threshold_slider.setValue(50)
        detect_layout.addWidget(self.threshold_slider, 0, 1)
        self.threshold_value = QLabel("50")
        self.threshold_slider.valueChanged.connect(
            lambda v: self.threshold_value.setText(str(v))
        )
        detect_layout.addWidget(self.threshold_value, 0, 2)
        
        detect_layout.addWidget(QLabel("Min Size (px):"), 1, 0)
        self.min_size_spin = QSpinBox()
        self.min_size_spin.setRange(1, 100)
        self.min_size_spin.setValue(3)
        detect_layout.addWidget(self.min_size_spin, 1, 1, 1, 2)
        
        detect_layout.addWidget(QLabel("Max Size (px):"), 2, 0)
        self.max_size_spin = QSpinBox()
        self.max_size_spin.setRange(1, 500)
        self.max_size_spin.setValue(50)
        detect_layout.addWidget(self.max_size_spin, 2, 1, 1, 2)
        
        params_layout.addWidget(detect_group)
        
        # Buttons
        params_layout.addStretch()
        
        preview_btn = QPushButton("👁️ Preview")
        preview_btn.setObjectName("primaryButton")
        params_layout.addWidget(preview_btn)
        
        apply_btn = QPushButton("✓ Apply to All")
        params_layout.addWidget(apply_btn)
        
        layout.addWidget(params_frame)
    
    def _browse_images(self):
        """Open file dialog to select images."""
        file_paths, _ = QFileDialog.getOpenFileNames(
            self,
            "Select Images",
            "",
            "Image Files (*.png *.jpg *.tif *.tiff *.bmp)"
        )
        if file_paths:
            self.image_count_label.setText(f"{len(file_paths)} images loaded")
