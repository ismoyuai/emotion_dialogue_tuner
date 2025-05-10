# main.py
"""主执行入口，协调各模块完成数据生成任务"""

from src.utils.config_loader import ConfigLoader
from src.utils.validator import ReplyValidator
from src.data_generator import StyleDataGenerator
import json
import os

def main():
    # 初始化配置加载器
    config_loader = ConfigLoader()

    # 加载配置信息
    try:
        settings = config_loader.load_settings()
        style_config = config_loader.load_style_config()
    except FileNotFoundError as e:
        print(f"配置文件缺失：{str(e)}")
        return

    # 初始化核心组件
    generator = StyleDataGenerator(
        api_key=settings["API"]["ZHIPU_API_KEY"],
        style_config=style_config
    )
    validator = ReplyValidator(
        model_path=settings["PATHS"]["EMBEDDING_MODEL"]
    )

    # 执行数据生成流程
    all_data = []
    try:
        print("正在生成温柔风格数据...")
        gentle_data = generator.generate_style_data("温柔", 2000)
        all_data.extend(gentle_data)

        print("正在生成毒舌风格数据...")
        sarcastic_data = generator.generate_style_data("毒舌", 2000)
        all_data.extend(sarcastic_data)

    except KeyboardInterrupt:
        print("\n用户中断操作，正在保存已生成数据...")
    finally:
        # 确保输出目录存在
        output_dir = "outputs"
        os.makedirs(output_dir, exist_ok=True)

        # 持久化保存数据
        output_path = os.path.join(output_dir, "style_chat_data.json")
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(all_data, f, ensure_ascii=False, indent=2)
        print(f"数据保存完成，有效样本数：{len(all_data)}")


if __name__ == "__main__":
    main()