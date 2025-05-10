# data_generator.py
"""数据生成核心模块，负责调用API生成指定风格的对话数据"""

from zhipuai import ZhipuAI
import random
import time

class StyleDataGenerator:
    """对话数据生成器，根据配置生成特定风格的对话数据"""

    def __init__(self, api_key: str, style_config: dict):
        """
        Args:
            api_key (str): 智普API访问密钥
            style_config (dict): 风格配置字典
        """
        self.client = ZhipuAI(api_key=api_key)
        self.style_config = style_config

    def _build_messages(self, style_name: str) -> list:
        """构建符合API要求的消息格式
        Args:
            style_name (str): 目标风格名称（如'温柔'）
        Returns:
            list: 包含系统提示和示例对话的消息列表
        """
        config = self.style_config[style_name]
        return [
            {"role": "system", "content": config["system_prompt"]},
            *config["examples"]  # 展开示例对话
        ]

    def generate_style_data(self, style_name: str, num_samples: int = 50) -> list:
        """生成指定风格的对话数据
        Args:
            style_name (str): 目标风格名称
            num_samples (int): 需要生成的样本数量
        Returns:
            list: 生成的对话数据列表，每个元素包含用户输入、助手回复和风格标签
        """
        data = []
        messages = self._build_messages(style_name)

        # 从本地文件加载用户输入
        user_inputs = []
        with open("data/cleaned_output.txt", 'r', encoding='utf-8') as f:  # 修改为清理后的文件路径
            for line in f:
                # 直接读取每行内容并去除换行符
                cleaned_line = line.rstrip('\n')  # 或使用 line.strip()
                if cleaned_line:  # 空行过滤（冗余保护）
                    user_inputs.append(cleaned_line)

        # 添加空值检查
        if not user_inputs:
            raise ValueError("文件内容为空或未成功加载数据，请检查："
                             "1. 文件路径是否正确 2. 文件是否包含有效内容")

        # 初始化顺序索引
        current_index = 0  # 添加索引计数器
        for _ in range(num_samples):
            try:

                # 按顺序选择用户输入（修改核心部分）
                user_msg = user_inputs[current_index]
                current_index = (current_index + 1) % len(user_inputs)  # 循环计数

                # 添加当前用户消息
                current_messages = messages + [{"role": "user", "content": user_msg}]

                # 调用大模型API生成回复
                response = self.client.chat.completions.create(
                    model="glm-3-turbo",
                    messages=current_messages,
                    temperature=self.style_config[style_name]["temperature"],
                    max_tokens=100
                )
                reply = response.choices[0].message.content

                # 保存通过质量检查的数据
                if self._validate_reply(style_name, user_msg, reply):
                    data.append({
                        "user": user_msg,
                        "assistant": reply,
                        "style": style_name
                    })

                time.sleep(0.5)  # API调用频率限制保护

            except Exception as e:
                print(f"生成失败: {str(e)}")

        return data

    def _validate_reply(self, style: str, user_msg: str, reply: str) -> bool:
        """内部方法：验证回复质量（实际实现应调用Validator类）"""
        # 简化的验证逻辑，实际应使用独立的Validator类
        return bool(reply)  # 示例代码