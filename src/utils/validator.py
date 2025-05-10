# validator.py
"""回复质量验证模块，确保生成数据符合质量标准"""

import numpy as np
from sentence_transformers import SentenceTransformer

class ReplyValidator:
    """回复验证器，执行多维度质量检查"""

    def __init__(self, model_path: str):
        """
        Args:
            model_path (str): 本地嵌入模型文件路径
        """
        self.style_model = SentenceTransformer(model_path)

    def validate(self, style: str, user_msg: str, reply: str, ref_text: str) -> bool:
        """执行完整的质量验证流程
        Args:
            style (str): 目标风格名称
            user_msg (str): 用户输入文本
            reply (str): 待验证的回复文本
            ref_text (str): 参考文本（用于相似度计算）
        Returns:
            bool: 是否通过所有验证规则
        """
        # 基础格式检查
        if not self._basic_checks(reply):
            print("内容为空或长度不够！")
            return False

        # 风格关键词匹配检查
        if not self._style_keyword_check(style, reply):
            print("不包含关键词！")
            return False

        # 语义相似度验证
        return self._semantic_similarity_check(ref_text, reply)

    def _basic_checks(self, reply: str) -> bool:
        """执行基础格式检查
        1. 非空检查
        2. 长度限制检查
        """
        return bool(reply) and (5 <= len(reply) <= 150)

    def _style_keyword_check(self, style: str, reply: str) -> bool:
        """检查是否包含风格特征关键词"""
        keyword_map = {
            "温柔": ["呢", "呀", "😊", "🌸"],
            "毒舌": ["好家伙", "栓Q", "!", "🏋️"]
        }
        return any(kw in reply for kw in keyword_map.get(style, []))

    def _semantic_similarity_check(self, ref_text: str, reply: str) -> bool:
        """计算与参考文本的语义相似度
        使用余弦相似度判断，阈值设为0.65
        """
        ref_vec = self.style_model.encode(ref_text)
        reply_vec = self.style_model.encode(reply)
        similarity = np.dot(ref_vec, reply_vec)
        print("======>similarity", similarity)
        return similarity > 0.65