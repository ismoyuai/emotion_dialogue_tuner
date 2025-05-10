# config_loader.py
"""配置文件加载模块，负责统一管理API密钥、模型路径等敏感信息和风格配置"""

import yaml
import json
from pathlib import Path


class ConfigLoader:
    """配置加载器，封装配置文件的读取操作"""

    def __init__(self):
        """初始化时自动定位项目根目录"""
        self.root_path = Path(__file__).resolve().parent.parent.parent  # 根据实际层级调整

    def load_settings(self) -> dict:
        """加载YAML格式的全局设置
        Returns:
            dict: 包含API密钥、模型路径等配置的字典
        """
        with open(self.root_path / "config/settings.yaml", "r", encoding="utf-8") as f:
            return yaml.safe_load(f)

    def load_style_config(self) -> dict:
        """加载JSON格式的风格配置
        Returns:
            dict: 包含不同对话风格的模板配置
        """
        with open(self.root_path / "config/style_config.json", "r", encoding="utf-8") as f:
            return json.load(f)