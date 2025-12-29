#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
MAT文件查看器 - 单文件Python客户端应用程序
支持查看MATLAB的.mat文件内容
"""

import sys
import os
import numpy as np
from pathlib import Path
from typing import Dict, Any, Optional

from PySide6.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
    QSplitter, QFileSystemModel, QTreeView, QListWidget, QListWidgetItem,
    QToolBar, QToolButton, QLabel, QStatusBar, QMessageBox, QDialog,
    QDialogButtonBox, QTextEdit, QTreeWidget, QTreeWidgetItem, QHeaderView,
    QProgressDialog, QFrame, QSizePolicy, QScrollArea, QGridLayout, QGroupBox,
    QPushButton, QLineEdit, QTableWidget, QTableWidgetItem, QAbstractItemView,
    QComboBox
)
from PySide6.QtCore import (
    Qt, QSize, QDir, QFileSystemWatcher, Signal, QObject, QThread,
    QModelIndex, QTimer
)
from PySide6.QtGui import (
    QIcon, QAction, QFont, QColor, QPalette, QPixmap, QPainter,
    QPainterPath, QLinearGradient, QCursor, QKeySequence, QShortcut
)

try:
    import scipy.io
    HAS_SCIPY = True
except ImportError:
    HAS_SCIPY = False

try:
    import mat73
    HAS_MAT73 = True
except ImportError:
    HAS_MAT73 = False


class MatFileParser(QObject):
    """MAT文件解析器"""
    
    parsing_complete = Signal(dict)
    error_occurred = Signal(str)
    
    def __init__(self):
        super().__init__()
        self.current_file = None
    
    def parse_file(self, file_path: str):
        """解析MAT文件"""
        try:
            data = {}
            file_path = str(file_path)
            
            try:
                mat_data = scipy.io.loadmat(file_path)
            except Exception as e:
                if HAS_MAT73:
                    try:
                        mat_data = mat73.loadmat(file_path)
                    except Exception:
                        raise ValueError(f"无法使用scipy或mat73解析文件: {e}")
                else:
                    raise ValueError(f"无法解析MAT文件 (scipy失败): {e}")
            
            for key, value in mat_data.items():
                if not key.startswith('__'):
                    data[key] = self._process_value(value)
            
            self.parsing_complete.emit(data)
            
        except Exception as e:
            self.error_occurred.emit(str(e))
    
    def _process_value(self, value, depth: int = 0) -> Dict[str, Any]:
        """处理值并生成描述信息"""
        if depth > 3:
            return {"type": "complex nested structure", "preview": "..."}
        
        result = {
            "type": type(value).__name__,
            "value": value
        }
        
        try:
            if isinstance(value, np.ndarray):
                result["shape"] = str(value.shape)
                result["dtype"] = str(value.dtype)
                
                if value.size <= 100:
                    try:
                        flat = value.flatten()
                        if all(isinstance(x, (int, float, np.integer, np.floating)) for x in flat[:min(10, len(flat))]):
                            sample_values = [float(x) for x in flat[:min(10, len(flat))]]
                            result["sample_values"] = sample_values
                            if len(flat) > 10:
                                result["sample_values"].append(float(np.mean(flat[10:])))
                    except:
                        pass
                        
            elif isinstance(value, (np.integer, np.floating, np.bool_)):
                result["dtype"] = str(type(value).__name__)
                result["value"] = float(value) if isinstance(value, (np.floating,)) else value
                
        except Exception:
            pass
        
        if hasattr(value, 'dtype') and value.dtype.names:
            result["is_struct"] = True
            result["fields"] = list(value.dtype.names)
        
        return result


class TableViewerDialog(QDialog):
    """表格查看器对话框"""
    
    def __init__(self, var_name: str, data: np.ndarray, parent=None):
        super().__init__(parent)
        self.var_name = var_name
        self.data = data
        self.page_size = 100
        self.current_page = 0
        self.total_pages = 0
        
        self.setup_ui()
        self.load_data()
    
    def setup_ui(self):
        """设置UI"""
        self.setWindowTitle(f"表格查看: {self.var_name}")
        self.setMinimumSize(900, 600)
        self.resize(1000, 700)
        
        layout = QVBoxLayout(self)
        layout.setContentsMargins(10, 10, 10, 10)
        layout.setSpacing(10)
        
        info_frame = QFrame()
        info_frame.setStyleSheet("""
            QFrame {
                background-color: #3A3A3A;
                border-radius: 5px;
                padding: 10px;
            }
        """)
        info_layout = QHBoxLayout(info_frame)
        
        shape = self.data.shape
        dtype = str(self.data.dtype)
        total_size = self.data.size
        
        info_label = QLabel(f"维度: {shape} | 类型: {dtype} | 总元素: {total_size:,}")
        info_label.setStyleSheet("""
            QLabel {
                color: #FFFFFF;
                font-size: 13px;
            }
        """)
        info_layout.addWidget(info_label)
        
        self.page_combo = QComboBox()
        self.page_combo.setStyleSheet("""
            QComboBox {
                background-color: #5A5A5A;
                color: #FFFFFF;
                padding: 5px 10px;
                border-radius: 3px;
                min-width: 80px;
            }
        """)
        self.page_combo.currentIndexChanged.connect(self.on_page_changed)
        info_layout.addWidget(self.page_combo)
        
        self.rows_label = QLabel()
        self.rows_label.setStyleSheet("""
            QLabel {
                color: #AAAAAA;
                font-size: 12px;
            }
        """)
        info_layout.addWidget(self.rows_label)
        
        info_layout.addStretch()
        
        self.prev_btn = QPushButton("◀ 上一页")
        self.prev_btn.setStyleSheet("""
            QPushButton {
                background-color: #5A5A5A;
                color: #FFFFFF;
                padding: 5px 15px;
                border-radius: 3px;
            }
            QPushButton:hover {
                background-color: #6A6A6A;
            }
            QPushButton:disabled {
                background-color: #4A4A4A;
                color: #888;
            }
        """)
        self.prev_btn.clicked.connect(self.prev_page)
        info_layout.addWidget(self.prev_btn)
        
        self.next_btn = QPushButton("下一页 ▶")
        self.next_btn.setStyleSheet("""
            QPushButton {
                background-color: #5A5A5A;
                color: #FFFFFF;
                padding: 5px 15px;
                border-radius: 3px;
            }
            QPushButton:hover {
                background-color: #6A6A6A;
            }
            QPushButton:disabled {
                background-color: #4A4A4A;
                color: #888;
            }
        """)
        self.next_btn.clicked.connect(self.next_page)
        info_layout.addWidget(self.next_btn)
        
        layout.addWidget(info_frame)
        
        self.table_widget = QTableWidget()
        self.table_widget.setStyleSheet("""
            QTableWidget {
                background-color: #E8E8E8;
                color: #333333;
                gridline-color: #B0B0B0;
                font-size: 12px;
            }
            QTableWidget::item {
                padding: 4px 8px;
            }
            QTableWidget::item:selected {
                background-color: #A0C4FF;
            }
            QTableWidget::item:hover {
                background-color: #D0D0D0;
            }
            QHeaderView::section {
                background-color: #B0B0B0;
                color: #333333;
                padding: 8px;
                font-weight: bold;
            }
            QHeaderView::section:hover {
                background-color: #C0C0C0;
            }
        """)
        self.table_widget.setAlternatingRowColors(True)
        self.table_widget.setSelectionBehavior(QAbstractItemView.SelectRows)
        self.table_widget.verticalHeader().setDefaultSectionSize(22)
        
        layout.addWidget(self.table_widget)
        
        button_box = QDialogButtonBox(QDialogButtonBox.Close)
        button_box.rejected.connect(self.reject)
        button_box.setStyleSheet("""
            QDialogButtonBox {
                background-color: transparent;
            }
            QPushButton {
                background-color: #5A5A5A;
                color: #FFFFFF;
                padding: 8px 20px;
                border-radius: 3px;
                min-width: 80px;
            }
            QPushButton:hover {
                background-color: #6A6A6A;
            }
        """)
        layout.addWidget(button_box)
    
    def load_data(self):
        """加载数据"""
        if self.data.ndim == 1:
            self.load_1d_data()
        elif self.data.ndim == 2:
            self.load_2d_data()
        else:
            self.load_nd_data()
    
    def load_1d_data(self):
        """加载1维数据"""
        total_rows = len(self.data)
        self.total_pages = (total_rows + self.page_size - 1) // self.page_size
        
        self.page_combo.clear()
        for i in range(self.total_pages):
            start_row = i * self.page_size
            end_row = min((i + 1) * self.page_size, total_rows)
            self.page_combo.addItem(f"行 {start_row + 1}-{end_row}")
        
        self.page_combo.setCurrentIndex(self.current_page)
        self.update_pagination_controls()
        self.display_page()
    
    def load_2d_data(self):
        """加载2维数据"""
        total_rows = self.data.shape[0]
        self.total_pages = (total_rows + self.page_size - 1) // self.page_size
        
        self.page_combo.clear()
        for i in range(self.total_pages):
            start_row = i * self.page_size
            end_row = min((i + 1) * self.page_size, total_rows)
            self.page_combo.addItem(f"行 {start_row + 1}-{end_row}")
        
        self.page_combo.setCurrentIndex(self.current_page)
        self.update_pagination_controls()
        self.display_page()
    
    def load_nd_data(self):
        """加载多维数据（重塑为2维）"""
        total_elements = self.data.size
        total_rows = min(total_elements, 10000)
        self.page_size = min(self.page_size, total_rows)
        self.total_pages = (total_rows + self.page_size - 1) // self.page_size
        
        self.page_combo.clear()
        for i in range(self.total_pages):
            start_row = i * self.page_size
            end_row = min((i + 1) * self.page_size, total_rows)
            self.page_combo.addItem(f"元素 {start_row + 1}-{end_row}")
        
        self.page_combo.setCurrentIndex(self.current_page)
        self.update_pagination_controls()
        self.display_page()
    
    def display_page(self):
        """显示当前页"""
        start_idx = self.current_page * self.page_size
        end_idx = min(start_idx + self.page_size, len(self.data.flatten()))
        
        if self.data.ndim == 1:
            self.display_1d_page(start_idx, end_idx)
        elif self.data.ndim == 2:
            self.display_2d_page(start_idx, end_idx)
        else:
            self.display_nd_page(start_idx, end_idx)
        
        start_row = start_idx + 1
        end_row = end_idx
        self.rows_label.setText(f"显示元素 {start_row:,} - {end_row:,} (共 {self.data.size:,} 个)")
    
    def display_1d_page(self, start_idx: int, end_idx: int):
        """显示1维数据页"""
        page_data = self.data[start_idx:end_idx]
        
        self.table_widget.setRowCount(len(page_data))
        self.table_widget.setColumnCount(2)
        self.table_widget.setHorizontalHeaderLabels(["索引", "值"])
        
        for row, (idx, value) in enumerate(zip(range(start_idx, end_idx), page_data)):
            idx_item = QTableWidgetItem(str(idx))
            idx_item.setForeground(QColor("#888888"))
            self.table_widget.setItem(row, 0, idx_item)
            
            value_item = QTableWidgetItem(self.format_value(value))
            self.table_widget.setItem(row, 1, value_item)
        
        self.table_widget.resizeColumnsToContents()
    
    def display_2d_page(self, start_idx: int, end_idx: int):
        """显示2维数据页"""
        page_data = self.data[start_idx:end_idx, :]
        num_cols = page_data.shape[1]
        
        self.table_widget.setRowCount(page_data.shape[0])
        self.table_widget.setColumnCount(num_cols)
        
        headers = [f"Col {i}" for i in range(num_cols)]
        self.table_widget.setHorizontalHeaderLabels(headers)
        
        for row_idx, row_data in enumerate(page_data):
            for col_idx, value in enumerate(row_data):
                item = QTableWidgetItem(self.format_value(value))
                self.table_widget.setItem(row_idx, col_idx, item)
        
        self.table_widget.resizeColumnsToContents()
    
    def display_nd_page(self, start_idx: int, end_idx: int):
        """显示多维数据页"""
        flat_data = self.data.flatten()
        page_data = flat_data[start_idx:end_idx]
        
        self.table_widget.setRowCount(len(page_data))
        self.table_widget.setColumnCount(2)
        self.table_widget.setHorizontalHeaderLabels(["元素索引", "值"])
        
        for row, (idx, value) in enumerate(zip(range(start_idx, end_idx), page_data)):
            idx_item = QTableWidgetItem(str(idx))
            idx_item.setForeground(QColor("#888888"))
            self.table_widget.setItem(row, 0, idx_item)
            
            value_item = QTableWidgetItem(self.format_value(value))
            self.table_widget.setItem(row, 1, value_item)
        
        self.table_widget.resizeColumnsToContents()
    
    def format_value(self, value) -> str:
        """格式化值"""
        try:
            if isinstance(value, (np.floating,)):
                return f"{float(value):.6g}"
            elif isinstance(value, (np.integer,)):
                return str(int(value))
            elif isinstance(value, (np.bool_,)):
                return "True" if value else "False"
            elif isinstance(value, bytes):
                return value.decode('utf-8', errors='replace')
            elif isinstance(value, np.ndarray):
                return f"Array({value.shape})"
            else:
                return str(value)
        except:
            return str(value)
    
    def update_pagination_controls(self):
        """更新分页控件状态"""
        self.prev_btn.setEnabled(self.current_page > 0)
        self.next_btn.setEnabled(self.current_page < self.total_pages - 1)
    
    def on_page_changed(self, index: int):
        """页码改变"""
        self.current_page = index
        self.display_page()
        self.update_pagination_controls()
    
    def prev_page(self):
        """上一页"""
        if self.current_page > 0:
            self.current_page -= 1
            self.page_combo.setCurrentIndex(self.current_page)
    
    def next_page(self):
        """下一页"""
        if self.current_page < self.total_pages - 1:
            self.current_page += 1
            self.page_combo.setCurrentIndex(self.current_page)


class CustomTitleBar(QWidget):
    """自定义标题栏"""
    
    def __init__(self, parent=None):
        super().__init__(parent)
        self.main_window = parent
        self.setFixedHeight(40)
        self.setup_ui()
        self.setup_connections()
    
    def setup_ui(self):
        layout = QHBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(0)
        
        title_label = QLabel("MAT文件查看器")
        title_label.setStyleSheet("""
            QLabel {
                color: #FFFFFF;
                font-size: 14px;
                font-weight: bold;
                padding-left: 15px;
            }
        """)
        layout.addWidget(title_label)
        
        layout.addStretch()
        
        button_layout = QHBoxLayout()
        button_layout.setSpacing(0)
        
        self.min_btn = self.create_window_button("─", self.minimize_window)
        self.max_btn = self.create_window_button("□", self.maximize_window)
        self.close_btn = self.create_window_button("✕", self.close_window)
        
        button_layout.addWidget(self.min_btn)
        button_layout.addWidget(self.max_btn)
        button_layout.addWidget(self.close_btn)
        
        container = QWidget()
        container.setLayout(button_layout)
        container.setStyleSheet("""
            QWidget {
                background-color: transparent;
            }
        """)
        layout.addWidget(container)
        
        self.setStyleSheet("""
            CustomTitleBar {
                background: qlineargradient(
                    x1:0, y1:0, x2:1, y2:0,
                    stop:0 #1a1a2e,
                    stop:1 #16213e
                );
                border-bottom: 1px solid #0f3460;
            }
        """)
    
    def create_window_button(self, text: str, callback):
        """创建窗口控制按钮"""
        button = QPushButton(text)
        button.setFixedSize(46, 40)
        button.setCursor(QCursor(Qt.PointingHandCursor))
        button.setStyleSheet("""
            QPushButton {
                background-color: transparent;
                color: #FFFFFF;
                font-size: 16px;
                border: none;
                padding: 0px;
            }
            QPushButton:hover {
                background-color: rgba(255, 255, 255, 0.1);
            }
            QPushButton#close_button:hover {
                background-color: #e81123;
            }
        """)
        button.clicked.connect(callback)
        return button
    
    def setup_connections(self):
        pass
    
    def minimize_window(self):
        if self.main_window:
            self.main_window.showMinimized()
    
    def maximize_window(self):
        if self.main_window:
            if self.main_window.isMaximized():
                self.main_window.showNormal()
                self.max_btn.setText("□")
            else:
                self.main_window.showMaximized()
                self.max_btn.setText("❐")
    
    def close_window(self):
        if self.main_window:
            self.main_window.close()


class FileIconProvider:
    """文件图标提供者"""
    
    @staticmethod
    def get_icon(file_path: str) -> QIcon:
        """根据文件类型返回图标"""
        if file_path.endswith('.mat'):
            return FileIconProvider._create_mat_icon()
        elif os.path.isdir(file_path):
            return FileIconProvider._create_folder_icon()
        else:
            return FileIconProvider._create_file_icon()
    
    @staticmethod
    def _create_mat_icon() -> QIcon:
        """创建MAT文件图标"""
        pixmap = QPixmap(32, 32)
        pixmap.fill(Qt.transparent)
        painter = QPainter(pixmap)
        painter.setRenderHint(QPainter.Antialiasing)
        
        painter.setPen(Qt.NoPen)
        painter.setBrush(QColor("#FF6B35"))
        painter.drawRoundedRect(4, 4, 24, 24, 4, 4)
        
        painter.setPen(QColor("#FFFFFF"))
        painter.setFont(QFont("Arial", 12, QFont.Bold))
        painter.drawText(pixmap.rect(), Qt.AlignCenter, "M")
        
        painter.end()
        return QIcon(pixmap)
    
    @staticmethod
    def _create_folder_icon() -> QIcon:
        """创建文件夹图标"""
        pixmap = QPixmap(32, 32)
        pixmap.fill(Qt.transparent)
        painter = QPainter(pixmap)
        painter.setRenderHint(QPainter.Antialiasing)
        
        painter.setPen(Qt.NoPen)
        painter.setBrush(QColor("#FFC107"))
        painter.drawRoundedRect(4, 8, 24, 20, 3, 3)
        painter.setBrush(QColor("#FFD54F"))
        painter.drawRoundedRect(4, 4, 14, 8, 3, 3)
        
        painter.end()
        return QIcon(pixmap)
    
    @staticmethod
    def _create_file_icon() -> QIcon:
        """创建通用文件图标"""
        pixmap = QPixmap(32, 32)
        pixmap.fill(Qt.transparent)
        painter = QPainter(pixmap)
        painter.setRenderHint(QPainter.Antialiasing)
        
        painter.setPen(Qt.NoPen)
        painter.setBrush(QColor("#9E9E9E"))
        painter.drawRoundedRect(6, 4, 20, 24, 2, 2)
        
        painter.setBrush(QColor("#757575"))
        painter.drawRect(10, 8, 12, 4)
        
        painter.end()
        return QIcon(pixmap)


class MatFileViewer(QMainWindow):
    """主窗口"""
    
    def __init__(self):
        super().__init__()
        self.current_directory = str(Path.cwd())
        self.current_mat_file = None
        self.mat_data = {}
        self.parser = MatFileParser()
        self.file_watcher = QFileSystemWatcher()
        
        self.setup_ui()
        self.setup_connections()
        self.load_directory(self.current_directory)
        
    def setup_ui(self):
        """设置UI"""
        self.setWindowTitle("MAT文件查看器")
        self.setMinimumSize(1200, 700)
        
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        
        main_layout = QVBoxLayout(central_widget)
        main_layout.setContentsMargins(0, 0, 0, 0)
        main_layout.setSpacing(0)
        
        main_layout.addWidget(self.create_toolbar())
        main_layout.addWidget(self.create_content_area())
        main_layout.addWidget(self.create_status_bar())
        
        self.apply_styles()
    
    def create_toolbar(self) -> QToolBar:
        """创建工具栏"""
        toolbar = QToolBar()
        toolbar.setMovable(False)
        toolbar.setIconSize(QSize(20, 20))
        toolbar.setFixedHeight(36)
        toolbar.setStyleSheet("""
            QToolBar {
                background-color: #1e1e2e;
                border-bottom: 1px solid #333;
                padding: 2px 5px;
                spacing: 5px;
            }
            QToolButton {
                background-color: transparent;
                color: #FFFFFF;
                padding: 4px 10px;
                border-radius: 3px;
                font-size: 13px;
            }
            QToolButton:hover {
                background-color: rgba(255, 255, 255, 0.1);
            }
            QToolButton:pressed {
                background-color: rgba(255, 255, 255, 0.2);
            }
        """)
        
        open_folder_action = QAction("📁 打开文件夹", self)
        open_folder_action.setStatusTip("打开文件夹")
        open_folder_action.triggered.connect(self.open_folder_dialog)
        toolbar.addAction(open_folder_action)
        
        open_file_action = QAction("📄 打开MAT文件", self)
        open_file_action.setStatusTip("打开单个MAT文件")
        open_file_action.triggered.connect(self.open_file_dialog)
        toolbar.addAction(open_file_action)
        
        toolbar.addSeparator()
        
        reload_action = QAction("🔄 刷新", self)
        reload_action.setStatusTip("刷新文件列表")
        reload_action.triggered.connect(self.reload_directory)
        toolbar.addAction(reload_action)
        
        self.path_line = QLineEdit()
        self.path_line.setPlaceholderText("当前路径...")
        self.path_line.setReadOnly(True)
        self.path_line.setStyleSheet("""
            QLineEdit {
                background-color: #2d2d3d;
                color: #FFFFFF;
                border: 1px solid #3d3d4d;
                border-radius: 5px;
                padding: 5px 10px;
                font-size: 12px;
            }
        """)
        self.path_line.setFixedWidth(300)
        
        toolbar.addWidget(self.path_line)
        
        return toolbar
    
    def create_content_area(self) -> QSplitter:
        """创建内容区域"""
        splitter = QSplitter(Qt.Horizontal)
        splitter.setHandleWidth(1)
        splitter.setStyleSheet("""
            QSplitter::handle {
                background-color: #333;
            }
        """)
        
        left_panel = self.create_file_list_panel()
        right_panel = self.create_mat_viewer_panel()
        
        splitter.addWidget(left_panel)
        splitter.addWidget(right_panel)
        splitter.setSizes([300, 900])
        
        return splitter
    
    def create_file_list_panel(self) -> QWidget:
        """创建文件列表面板"""
        panel = QFrame()
        panel.setFrameShape(QFrame.StyledPanel)
        panel.setStyleSheet("""
            QFrame {
                background-color: #1e1e2e;
                border-right: 1px solid #333;
            }
        """)
        
        layout = QVBoxLayout(panel)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(0)
        
        header = QLabel("文件列表")
        header.setStyleSheet("""
            QLabel {
                color: #FFFFFF;
                font-size: 14px;
                font-weight: bold;
                padding: 10px 15px;
                background-color: #252535;
                border-bottom: 1px solid #333;
            }
        """)
        layout.addWidget(header)
        
        self.file_list = QListWidget()
        self.file_list.setStyleSheet("""
            QListWidget {
                background-color: #1e1e2e;
                color: #FFFFFF;
                border: none;
                padding: 5px;
            }
            QListWidget::item {
                padding: 8px 10px;
                border-radius: 4px;
                margin: 2px 5px;
            }
            QListWidget::item:selected {
                background-color: #4a4a6a;
                color: #FFFFFF;
            }
            QListWidget::item:hover {
                background-color: #2a2a4a;
            }
        """)
        self.file_list.itemClicked.connect(self.on_file_clicked)
        layout.addWidget(self.file_list)
        
        return panel
    
    def create_mat_viewer_panel(self) -> QWidget:
        """创建MAT文件查看面板"""
        panel = QWidget()
        panel.setStyleSheet("""
            QWidget {
                background-color: #252535;
            }
        """)
        
        layout = QVBoxLayout(panel)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(0)
        
        self.header_frame = QFrame()
        self.header_frame.setStyleSheet("""
            QFrame {
                background-color: #2a2a3a;
                border-bottom: 1px solid #333;
            }
        """)
        header_layout = QHBoxLayout(self.header_frame)
        
        self.mat_file_label = QLabel("未选择文件")
        self.mat_file_label.setStyleSheet("""
            QLabel {
                color: #FFFFFF;
                font-size: 16px;
                font-weight: bold;
                padding: 15px;
            }
        """)
        header_layout.addWidget(self.mat_file_label)
        
        header_layout.addStretch()
        
        info_label = QLabel("变量信息")
        info_label.setStyleSheet("""
            QLabel {
                color: #888;
                font-size: 12px;
                padding-right: 15px;
            }
        """)
        header_layout.addWidget(info_label)
        
        layout.addWidget(self.header_frame)
        
        self.mat_content_tree = QTreeWidget()
        self.mat_content_tree.setStyleSheet("""
            QTreeWidget {
                background-color: #252535;
                color: #FFFFFF;
                border: none;
                font-size: 13px;
            }
            QTreeWidget::item {
                padding: 8px 15px;
                border-bottom: 1px solid #333;
            }
            QTreeWidget::item:selected {
                background-color: #3a3a5a;
            }
            QTreeWidget::item:hover {
                background-color: #2a2a4a;
            }
            QTreeWidget::branch {
                background-color: #252535;
            }
            QHeaderView::section {
                background-color: #2a2a3a;
                color: #FFFFFF;
                padding: 10px;
                border: none;
                font-weight: bold;
            }
        """)
        self.mat_content_tree.setHeaderLabels(["名称", "类型", "形状/大小", "示例值"])
        self.mat_content_tree.header().setSectionResizeMode(0, QHeaderView.ResizeToContents)
        self.mat_content_tree.header().setSectionResizeMode(1, QHeaderView.ResizeToContents)
        self.mat_content_tree.header().setSectionResizeMode(2, QHeaderView.ResizeToContents)
        self.mat_content_tree.header().setSectionResizeMode(3, QHeaderView.Stretch)
        self.mat_content_tree.itemDoubleClicked.connect(self.on_tree_item_double_clicked)
        
        layout.addWidget(self.mat_content_tree)
        
        return panel
    
    def create_status_bar(self) -> QFrame:
        """创建状态栏"""
        status_bar = QFrame()
        status_bar.setFixedHeight(12)
        status_bar.setSizePolicy(QSizePolicy.Preferred, QSizePolicy.Fixed)
        status_bar.setStyleSheet("""
            QFrame {
                background-color: #1e1e2e;
                color: #888;
                border-top: 1px solid #333;
            }
        """)
        
        layout = QHBoxLayout(status_bar)
        layout.setContentsMargins(5, 1, 5, 1)
        layout.setSpacing(0)
        
        status_label = QLabel("就绪")
        status_label.setStyleSheet("""
            QLabel {
                color: #888;
                font-size: 10px;
            }
        """)
        layout.addWidget(status_label)
        
        layout.addStretch()
        
        return status_bar
    
    def apply_styles(self):
        """应用全局样式"""
        self.setStyleSheet("""
            QMainWindow {
                background-color: #1a1a2e;
            }
            QMenuBar {
                background-color: #1e1e2e;
                color: #FFFFFF;
                border-bottom: 1px solid #333;
            }
            QMenuBar::item:selected {
                background-color: #3a3a5a;
            }
            QMenu {
                background-color: #2a2a3a;
                color: #FFFFFF;
                border: 1px solid #444;
            }
            QMenu::item:selected {
                background-color: #4a4a6a;
            }
        """)
    
    def setup_connections(self):
        """设置信号连接"""
        self.parser.parsing_complete.connect(self.on_parsing_complete)
        self.parser.error_occurred.connect(self.on_parsing_error)
        self.file_watcher.directoryChanged.connect(self.on_directory_changed)
    
    def open_folder_dialog(self):
        """打开文件夹对话框"""
        from PySide6.QtWidgets import QFileDialog
        folder = QFileDialog.getExistingDirectory(
            self,
            "选择文件夹",
            self.current_directory,
            QFileDialog.ShowDirsOnly
        )
        if folder:
            self.load_directory(folder)
    
    def open_file_dialog(self):
        """打开文件对话框"""
        from PySide6.QtWidgets import QFileDialog
        files, _ = QFileDialog.getOpenFileNames(
            self,
            "选择MAT文件",
            self.current_directory,
            "MAT文件 (*.mat);;所有文件 (*.*)"
        )
        if files:
            self.load_mat_file(files[0])
    
    def load_directory(self, path: str):
        """加载目录"""
        self.current_directory = path
        self.path_line.setText(path)
        
        if not self.file_watcher.directories():
            self.file_watcher.addPath(path)
        
        self.file_list.clear()
        
        try:
            dir_path = Path(path)
            all_items = sorted(dir_path.iterdir(), key=lambda x: (not x.is_dir(), x.name.lower()))
            
            for item in all_items:
                icon = FileIconProvider.get_icon(str(item))
                list_item = QListWidgetItem(icon, item.name)
                list_item.setData(Qt.UserRole, str(item))
                
                if item.suffix.lower() == '.mat':
                    font = list_item.font()
                    font.setBold(True)
                    list_item.setFont(font)
                    list_item.setForeground(QColor("#FF6B35"))
                
                self.file_list.addItem(list_item)
            
            self.statusBar().showMessage(f"已加载 {self.file_list.count()} 个项目")
            
        except Exception as e:
            QMessageBox.warning(self, "错误", f"无法加载目录:\n{str(e)}")
    
    def reload_directory(self):
        """刷新目录"""
        self.load_directory(self.current_directory)
        self.statusBar().showMessage("已刷新")
    
    def on_directory_changed(self, path):
        """目录变化事件"""
        QTimer.singleShot(500, lambda: self.load_directory(path))
    
    def on_file_clicked(self, item: QListWidgetItem):
        """文件点击事件"""
        file_path = item.data(Qt.UserRole)
        path = Path(file_path)
        
        if path.is_file() and path.suffix.lower() == '.mat':
            self.load_mat_file(file_path)
        elif path.is_dir():
            self.load_directory(file_path)
    
    def load_mat_file(self, file_path: str):
        """加载MAT文件"""
        self.current_mat_file = file_path
        file_name = os.path.basename(file_path)
        self.mat_file_label.setText(f"📄 {file_name}")
        
        self.mat_content_tree.clear()
        
        loading_item = QTreeWidgetItem(self.mat_content_tree, ["正在加载...", "", "", ""])
        self.mat_content_tree.addTopLevelItem(loading_item)
        
        self.statusBar().showMessage(f"正在解析 {file_name}...")
        
        self.parser.parse_file(file_path)
    
    def on_parsing_complete(self, data: dict):
        """解析完成"""
        self.mat_content_tree.clear()
        self.mat_data = data
        
        if not data:
            no_data_item = QTreeWidgetItem(self.mat_content_tree, ["无数据", "", "", ""])
            self.mat_content_tree.addTopLevelItem(no_data_item)
            self.statusBar().showMessage("文件为空或无可读数据")
            return
        
        for var_name, var_info in data.items():
            self.add_tree_item(None, var_name, var_info)
        
        self.mat_content_tree.expandAll()
        self.statusBar().showMessage(f"已加载 {len(data)} 个变量")
    
    def on_parsing_error(self, error: str):
        """解析错误"""
        self.mat_content_tree.clear()
        error_item = QTreeWidgetItem(self.mat_content_tree, ["错误", "", "", ""])
        error_item.setText(1, "无法解析文件")
        error_item.setText(3, error)
        self.mat_content_tree.addTopLevelItem(error_item)
        
        self.statusBar().showMessage("解析失败")
        QMessageBox.critical(self, "解析错误", f"无法解析MAT文件:\n{error}")
    
    def add_tree_item(self, parent: Optional[QTreeWidgetItem], name: str, data: dict):
        """添加树节点"""
        item_type = data.get("type", "unknown")
        shape = data.get("shape", "")
        sample_values = data.get("sample_values", [])
        
        sample_str = ""
        if sample_values:
            if len(sample_values) <= 5:
                sample_str = ", ".join([f"{v:.4g}" for v in sample_values])
            else:
                sample_str = f"{', '.join([f'{v:.4g}' for v in sample_values[:3]])}, ... mean={sample_values[-1]:.4g}"
        
        if parent:
            item = QTreeWidgetItem(parent, [name, item_type, shape, sample_str])
        else:
            item = QTreeWidgetItem(self.mat_content_tree, [name, item_type, shape, sample_str])
            self.mat_content_tree.addTopLevelItem(item)
        
        if data.get("is_struct"):
            item.setExpanded(True)
        
        item.setData(0, Qt.UserRole, data)
        
        return item
    
    def on_tree_item_double_clicked(self, item: QTreeWidgetItem, column: int):
        """树节点双击事件 - 打开表格查看器"""
        var_name = item.text(0)
        var_info = item.data(0, Qt.UserRole)
        
        if not var_info:
            return
        
        value = var_info.get("value")
        if value is None:
            return
        
        if isinstance(value, np.ndarray):
            var_type = var_info.get("type", "")
            if var_type in ["ndarray", "matrix"]:
                dialog = TableViewerDialog(var_name, value, self)
                dialog.exec()
            else:
                QMessageBox.information(
                    self, 
                    "提示", 
                    f"变量类型为 '{var_type}'，暂不支持表格化查看"
                )
        else:
            QMessageBox.information(
                self, 
                "提示", 
                f"变量类型为 '{type(value).__name__}'，暂不支持表格化查看"
            )
    
    def closeEvent(self, event):
        """关闭事件"""
        event.accept()


def main():
    """主函数"""
    app = QApplication(sys.argv)
    app.setStyle("Fusion")
    
    window = MatFileViewer()
    window.show()
    
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
