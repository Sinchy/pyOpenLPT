"""
Custom Range Slider Widget with two visual draggable handles.
"""
from PySide6.QtWidgets import QWidget, QHBoxLayout, QLabel
from PySide6.QtCore import Qt, Signal, QPoint
from PySide6.QtGui import QPainter, QColor, QPen, QLinearGradient


class SimpleSlider(QWidget):
    """A single-handle slider with value label for float values like sensitivity."""
    
    valueChanged = Signal(float)
    
    def __init__(self, min_val=0.0, max_val=1.0, initial=0.5, decimals=2, parent=None):
        super().__init__(parent)
        
        self._min = min_val
        self._max = max_val
        self._value = initial
        self._decimals = decimals
        self._handle_radius = 7
        self._track_height = 5
        self._dragging = False
        
        self.setMinimumHeight(30)
        self.setCursor(Qt.CursorShape.PointingHandCursor)
        self.setMouseTracking(True)
        
        layout = QHBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(5)
        
        self._canvas = _SimpleSliderCanvas(self)
        
        self._label = QLabel(f"{initial:.{decimals}f}")
        self._label.setFixedWidth(35)
        self._label.setAlignment(Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignVCenter)
        self._label.setStyleSheet("color: #00d4ff; font-size: 10px;")
        
        layout.addWidget(self._canvas, 1)
        layout.addWidget(self._label)
    
    def value(self):
        return self._value
    
    def setValue(self, val):
        self._value = max(self._min, min(self._max, val))
        self._label.setText(f"{self._value:.{self._decimals}f}")
        self._canvas.update()
        self.valueChanged.emit(self._value)


class _SimpleSliderCanvas(QWidget):
    """Canvas for SimpleSlider."""
    
    def __init__(self, parent_slider):
        super().__init__(parent_slider)
        self._parent = parent_slider
        self._dragging = False
        self.setMouseTracking(True)
        self.setMinimumWidth(80)
    
    def _val_to_x(self, val):
        margin = self._parent._handle_radius
        width = self.width() - 2 * margin
        ratio = (val - self._parent._min) / (self._parent._max - self._parent._min)
        return int(margin + ratio * width)
    
    def _x_to_val(self, x):
        margin = self._parent._handle_radius
        width = self.width() - 2 * margin
        ratio = (x - margin) / width
        ratio = max(0, min(1, ratio))
        return self._parent._min + ratio * (self._parent._max - self._parent._min)
    
    def paintEvent(self, event):
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)
        
        h = self.height()
        w = self.width()
        r = self._parent._handle_radius
        track_h = self._parent._track_height
        
        val_x = self._val_to_x(self._parent._value)
        track_y = h // 2 - track_h // 2
        
        # Draw track background
        painter.setPen(Qt.PenStyle.NoPen)
        painter.setBrush(QColor(60, 70, 80))
        painter.drawRoundedRect(r, track_y, w - 2*r, track_h, track_h//2, track_h//2)
        
        # Draw filled portion
        gradient = QLinearGradient(r, 0, val_x, 0)
        gradient.setColorAt(0, QColor(0, 180, 220))
        gradient.setColorAt(1, QColor(0, 220, 255))
        painter.setBrush(gradient)
        painter.drawRoundedRect(r, track_y, val_x - r, track_h, track_h//2, track_h//2)
        
        # Draw handle
        painter.setBrush(QColor(0, 212, 255))
        painter.setPen(QPen(QColor(255, 255, 255), 2))
        painter.drawEllipse(QPoint(val_x, h // 2), r, r)
        
        painter.end()
    
    def mousePressEvent(self, event):
        x = event.pos().x()
        val_x = self._val_to_x(self._parent._value)
        r = self._parent._handle_radius
        
        if abs(x - val_x) <= r + 5 or True:  # Click anywhere moves handle
            self._dragging = True
            self._update_value(x)
    
    def mouseMoveEvent(self, event):
        if self._dragging:
            self._update_value(event.pos().x())
    
    def mouseReleaseEvent(self, event):
        self._dragging = False
    
    def _update_value(self, x):
        val = self._x_to_val(x)
        self._parent._value = round(val, self._parent._decimals)
        self._parent._label.setText(f"{self._parent._value:.{self._parent._decimals}f}")
        self.update()
        self._parent.valueChanged.emit(self._parent._value)


class RangeSlider(QWidget):
    """A visual slider with two draggable handles for min/max range selection."""
    
    rangeChanged = Signal(int, int)
    
    def __init__(self, min_val=1, max_val=500, initial_min=20, initial_max=200, suffix="", parent=None):
        super().__init__(parent)
        
        self._min_range = min_val
        self._max_range = max_val
        self._min_val = initial_min
        self._max_val = initial_max
        self._suffix = suffix
        
        self._handle_radius = 8
        self._track_height = 6
        self._dragging_min = False
        self._dragging_max = False
        
        self.setMinimumHeight(30)
        self.setCursor(Qt.CursorShape.PointingHandCursor)
        
        # Layout: [min_label] [slider] [max_label]
        layout = QHBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(3)
        
        # Narrow labels to minimize misalignment
        self.min_label = QLabel(f"{initial_min}")
        self.min_label.setFixedWidth(25)
        self.min_label.setAlignment(Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter)
        self.min_label.setStyleSheet("color: #00d4ff; font-size: 10px;")
        
        self.max_label = QLabel(f"{initial_max}")
        self.max_label.setFixedWidth(25)
        self.max_label.setAlignment(Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignVCenter)
        self.max_label.setStyleSheet("color: #00d4ff; font-size: 10px;")
        
        self.slider_canvas = _SliderCanvas(self)
        
        layout.addWidget(self.min_label)
        layout.addWidget(self.slider_canvas, 1)
        layout.addWidget(self.max_label)
    
    def _update_label(self):
        self.min_label.setText(str(self._min_val))
        self.max_label.setText(str(self._max_val))
    
    def value(self):
        return (self._min_val, self._max_val)
    
    def minValue(self):
        return self._min_val
    
    def maxValue(self):
        return self._max_val
    
    def setMinValue(self, val):
        self._min_val = max(self._min_range, min(val, self._max_val))
        self._update_label()
        self.slider_canvas.update()
        self.rangeChanged.emit(self._min_val, self._max_val)
    
    def setMaxValue(self, val):
        self._max_val = min(self._max_range, max(val, self._min_val))
        self._update_label()
        self.slider_canvas.update()
        self.rangeChanged.emit(self._min_val, self._max_val)


class _SliderCanvas(QWidget):
    """Internal canvas for drawing the slider track and handles."""
    
    def __init__(self, parent_slider):
        super().__init__(parent_slider)
        self._parent = parent_slider
        self._dragging_min = False
        self._dragging_max = False
        self.setMouseTracking(True)
        self.setMinimumWidth(80)
    
    def _val_to_x(self, val):
        margin = self._parent._handle_radius
        width = self.width() - 2 * margin
        ratio = (val - self._parent._min_range) / (self._parent._max_range - self._parent._min_range)
        return int(margin + ratio * width)
    
    def _x_to_val(self, x):
        margin = self._parent._handle_radius
        width = self.width() - 2 * margin
        ratio = (x - margin) / width
        ratio = max(0, min(1, ratio))
        return int(self._parent._min_range + ratio * (self._parent._max_range - self._parent._min_range))
    
    def paintEvent(self, event):
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)
        
        h = self.height()
        w = self.width()
        r = self._parent._handle_radius
        track_h = self._parent._track_height
        
        min_x = self._val_to_x(self._parent._min_val)
        max_x = self._val_to_x(self._parent._max_val)
        
        track_y = h // 2 - track_h // 2
        
        # Draw track background
        painter.setPen(Qt.PenStyle.NoPen)
        painter.setBrush(QColor(60, 70, 80))
        painter.drawRoundedRect(r, track_y, w - 2*r, track_h, track_h//2, track_h//2)
        
        # Draw selected range
        gradient = QLinearGradient(min_x, 0, max_x, 0)
        gradient.setColorAt(0, QColor(0, 180, 220))
        gradient.setColorAt(1, QColor(0, 220, 255))
        painter.setBrush(gradient)
        painter.drawRoundedRect(min_x, track_y, max_x - min_x, track_h, track_h//2, track_h//2)
        
        # Draw handles
        painter.setBrush(QColor(0, 212, 255))
        painter.setPen(QPen(QColor(255, 255, 255), 2))
        painter.drawEllipse(QPoint(min_x, h // 2), r, r)
        painter.drawEllipse(QPoint(max_x, h // 2), r, r)
        
        painter.end()
    
    def mousePressEvent(self, event):
        x = event.pos().x()
        min_x = self._val_to_x(self._parent._min_val)
        max_x = self._val_to_x(self._parent._max_val)
        r = self._parent._handle_radius
        
        if abs(x - min_x) <= r + 5:
            self._dragging_min = True
        elif abs(x - max_x) <= r + 5:
            self._dragging_max = True
        elif min_x < x < max_x:
            if x - min_x < max_x - x:
                self._dragging_min = True
            else:
                self._dragging_max = True
    
    def mouseMoveEvent(self, event):
        x = event.pos().x()
        val = self._x_to_val(x)
        
        if self._dragging_min:
            new_min = min(val, self._parent._max_val - 1)
            self._parent._min_val = max(self._parent._min_range, new_min)
            self._parent._update_label()
            self.update()
            self._parent.rangeChanged.emit(self._parent._min_val, self._parent._max_val)
        elif self._dragging_max:
            new_max = max(val, self._parent._min_val + 1)
            self._parent._max_val = min(self._parent._max_range, new_max)
            self._parent._update_label()
            self.update()
            self._parent.rangeChanged.emit(self._parent._min_val, self._parent._max_val)
    
    
    def mouseReleaseEvent(self, event):
        self._dragging_min = False
        self._dragging_max = False


from PySide6.QtWidgets import QDialog, QVBoxLayout, QPushButton, QProgressBar
from PySide6.QtCore import Qt

class ProcessingDialog(QDialog):
    """Dialog to show progress and allow Pause/Resume/Stop."""
    
    pause_signal = Signal(bool) # True = Pause, False = Resume
    stop_signal = Signal()
    
    def __init__(self, parent=None, title="Processing..."):
        super().__init__(parent)
        self.setWindowTitle(title)
        self.setFixedSize(400, 150)
        self.setWindowModality(Qt.WindowModality.ApplicationModal)
        self.setWindowFlags(self.windowFlags() & ~Qt.WindowType.WindowCloseButtonHint) # Disable close button
        
        layout = QVBoxLayout(self)
        
        self.status_label = QLabel("Initializing...")
        self.status_label.setStyleSheet("color: white; font-size: 14px;")
        layout.addWidget(self.status_label)
        
        self.progress_bar = QProgressBar()
        self.progress_bar.setStyleSheet("""
            QProgressBar {
                border: 1px solid #555;
                border-radius: 5px;
                background-color: #222;
                text-align: center;
                color: white;
            }
            QProgressBar::chunk {
                background-color: #00d4ff;
                width: 10px;
            }
        """)
        layout.addWidget(self.progress_bar)
        
        # Buttons
        btn_layout = QHBoxLayout()
        
        self.btn_pause = QPushButton("Pause")
        self.btn_pause.setCheckable(True)
        self.btn_pause.setStyleSheet("background-color: #e6b800; color: black; font-weight: bold; padding: 5px;")
        self.btn_pause.clicked.connect(self._toggle_pause)
        
        self.btn_stop = QPushButton("Stop")
        self.btn_stop.setStyleSheet("background-color: #d93025; color: white; font-weight: bold; padding: 5px;")
        self.btn_stop.clicked.connect(self._stop_processing)
        
        btn_layout.addWidget(self.btn_pause)
        btn_layout.addWidget(self.btn_stop)
        
        layout.addLayout(btn_layout)
        
        self._is_paused = False
        
    def update_progress(self, current, total):
        self.progress_bar.setMaximum(total)
        self.progress_bar.setValue(current)
        self.status_label.setText(f"Processing Frame: {current}/{total}")
        
    def _toggle_pause(self):
        self._is_paused = not self._is_paused
        self.btn_pause.setText("Resume" if self._is_paused else "Pause")
        self.status_label.setText("Paused" if self._is_paused else self.status_label.text())
        self.pause_signal.emit(self._is_paused)
        
    def _stop_processing(self):
        self.status_label.setText("Stopping...")
        self.btn_stop.setEnabled(False)
        self.btn_pause.setEnabled(False)
        self.stop_signal.emit()
