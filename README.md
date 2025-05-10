
# 【项目实战】大模型微调-情绪对话模型

[![Author](https://img.shields.io/badge/Author-墨宇Logic-blue.svg)](https://blog.ismoyu.cn)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE) <!-- 请确保项目中包含LICENSE文件 -->
![Python](https://img.shields.io/badge/Python-3.10+-blue?logo=python&logoColor=white)
![PyTorch](https://img.shields.io/badge/PyTorch-2.x-orange?logo=pytorch&logoColor=white)
![XTuner](https://img.shields.io/badge/Framework-XTuner-brightgreen)
![LMDeploy](https://img.shields.io/badge/Deployment-LMDeploy-purple)
![Streamlit](https://img.shields.io/badge/Frontend-Streamlit-ff69b4)

## 📖 项目简介与目标

本项目旨在带领学习者完成一个**具备强烈和个性化情绪表达能力**的对话大模型的微调全流程。灵感来源于抖音上的“小智聊天机器人”，目标是构建一个能在对话中展现特定情绪（如“温柔”、“毒舌”）的模型。项目覆盖从数据准备、模型选型、使用XTuner进行训练、评估到最终使用LMDeploy和Streamlit进行部署与交互测试的完整链路。

**核心目标：**

*   构建高质量、特定情绪风格的对话数据集。
*   选择合适的预训练模型（本项目选用Qwen1.5-1.8B-Chat）并微调以增强其情感表达能力。
*   掌握大模型微调的核心技术和工具链（XTuner, OpenCompass, LMDeploy）。
*   实现模型的高效部署和交互式测试。
*   探索模型在心理辅服、电商客服等场景的应用潜力。

## ✨ 项目特性

*   **端到端实践：** 完整覆盖LLM微调的各个关键环节，提供宝贵的动手经验。
*   **问题驱动：** 以解决实际应用中情绪化表达的需求为出发点。
*   **实用工具链：** 选用业界主流的XTuner、OpenCompass、LMDeploy等工具。
*   **重视数据质量：** 详细介绍AI辅助生成高质量数据集及多维度验证方法。
*   **迭代优化：** 记录训练过程中的问题、解决方案及多次训练的经验总结。
*   **情绪可定制：** 通过配置文件定义和生成不同风格（如温柔、毒舌）的对话数据。

## 🛠️ 技术栈

| 组件 (Component)     | 技术/工具 (Technology/Tool)                   | 角色/用途 (Role/Purpose)                                     |
| :------------------- | :-------------------------------------------- | :----------------------------------------------------------- |
| 编程语言 (Language)  | Python 3.10+                                  | 主要开发语言                                                 |
| 数据生成 (Data Gen)  | ZhipuAI API (glm-3-turbo)                     | AI辅助生成情绪对话数据                                       |
| 文本嵌入 (Embeddings) | `thomas/text2vec-base-chinese`                | 用于语义相似度校验                                           |
| 模型评估 (Evaluation) | OpenCompass                                   | 评估候选模型的中文理解能力 (CLUE数据集)                      |
| 模型微调 (Fine-tuning) | XTuner                                        | 主要的微调框架，支持QLoRA                                    |
| 模型部署 (Deployment) | LMDeploy                                      | 大模型推理框架，用于服务化部署                               |
| 前端交互 (Frontend)   | Streamlit                                     | 快速搭建交互式Web应用，用于模型效果测试                      |
| 配置管理 (Config)    | YAML, JSON                                    | 存储API密钥、路径、风格模板等配置                            |
| 操作系统 (OS)        | Windows11 + WSL2 (Ubuntu 22.04)               | 开发与训练环境 (推荐Linux环境)                               |
| 硬件 (Hardware)      | GeForce RTX 4060 Ti 16GB (或同等级显卡)       | 模型训练硬件                                                 |

## 📂 项目结构

```
emotion_dialogue_tuner/
├── config/                         # 配置文件
│   ├── settings.yaml               # API密钥等敏感配置
│   └── style_config.json           # 风格模板配置
├── data/
│   ├── LCCC-base-split/            # 公开原始数据集
│   ├── cleaned_output.txt          # 处理后用户对话数据
├── embedding_model/                # embedding模型路径
├── outputs/                        # 生成数据保存目录
├── src/
│   ├── data_generator.py           # 数据生成核心逻辑
│   ├── utils/
│   │   ├── chat_template_utils.py  # 模型对话模板转换模块
│   │   ├── config_loader.py        # 配置加载模块
│   │   ├── data_convert.py         # 训练数据格式转换模块
│   │   ├── data_extraction.py      # 提取清洗数据模块
│   │   └── validator.py            # 回复质量校验
└── main.py                         # 主执行入口
```

## 🚀 环境准备与安装指南

1.  **基础环境：**
    *   推荐使用 Linux (如 Ubuntu 22.04) 或 Windows 11 + WSL2 (Ubuntu 22.04)。
    *   NVIDIA 显卡，并安装对应的 [NVIDIA驱动](https://www.nvidia.com/Download/index.aspx) 和 [CUDA Toolkit](https://developer.nvidia.com/cuda-toolkit-archive) (教程中提及CUDA 12.1，请根据XTuner和PyTorch版本兼容性选择)。

2.  **Conda环境 (推荐)：**
    ```bash
    conda create -n emotion_llm_env python=3.10 -y
    conda activate emotion_llm_env
    ```

3.  **安装核心依赖：**
    *   **PyTorch:** 根据您的CUDA版本从 [PyTorch官网](https://pytorch.org/get-started/locally/) 获取安装命令。教程中因版本兼容问题，最终使用了：
        ```bash
        pip install torch==2.2.0+cu121 torchvision==0.17.0+cu121 --extra-index-url https://download.pytorch.org/whl/cu121
        ```
    *   **XTuner:**
        ```bash
        # 克隆XTuner仓库 (如果需要从源码安装或修改)
        # git clone https://github.com/InternLM/xtuner.git
        # cd xtuner
        # pip install -e '.[all]'

        # 或者直接pip安装 (可能版本与教程有差异)
        pip install xtuner[all] -U
        ```
        *注意：教程中遇到`bitsandbytes`和`triton.ops`的依赖问题，通过调整PyTorch和CUDA版本，并重新安装XTuner依赖解决。请务必关注XTuner官方文档关于环境配置的说明。*

    *   **LMDeploy:**
        ```bash
        pip install lmdeploy
        ```
    *   **OpenCompass (可选，用于模型评估):**
        ```bash
        # 克隆OpenCompass仓库
        # git clone https://github.com/open-compass/opencompass.git
        # cd opencompass
        # pip install -r requirements.txt
        # python setup.py develop
        ```
    *   **Streamlit 和其他:**
        ```bash
        pip install streamlit openai zhipuai pyyaml sentence_transformers numpy
        ```
    *   **项目依赖:**
        克隆本项目后，可以创建`requirements.txt`文件并使用`pip install -r requirements.txt`安装。

4.  **下载预训练模型和嵌入模型：**
    *   **Qwen1.5-1.8B-Chat:** 从Hugging Face Hub或ModelScope下载，并放置到指定路径 (如 `llm/Qwen/Qwen1___5-1___8B-Chat`)。
    *   **text2vec-base-chinese:** 从ModelScope (`thomas/text2vec-base-chinese`)下载，并放置到指定路径 (如 `embedding_model/thomas/text2vec-base-chinese`)。

## 📊 数据准备步骤

1.  **收集原始问题：**
    *   从 [CDial-GPT](https://github.com/thu-coai/CDial-GPT) 或 魔塔社区 [LCCC](https://www.modelscope.cn/datasets/OmniData/LCCC) 等数据集中提取对话的问题部分。
    *   将这些问题整理到一个文本文件中，例如 `data/cleaned_output.txt`，每行一个问题。

2.  **配置API密钥和路径 (`config/settings.yaml`)：**
    ```yaml
    API:
      ZHIPU_API_KEY: "your_zhipu_api_key_here"  # 替换为你的智谱AI API Key
      MODEL_NAME: "glm-3-turbo"

    PATHS:
      EMBEDDING_MODEL: "path/to/your/embedding_model/thomas/text2vec-base-chinese" # 替换为你的嵌入模型本地路径
    ```

3.  **配置对话风格 (`config/style_config.json`)：**
    定义AI生成数据时应遵循的系统提示和对话示例。
    ```json
    {
        "温柔": {
            "system_prompt": "你是一个温柔体贴的聊天助手...",
            "examples": [
                {"role": "user", "content": "今天好累啊"},
                {"role": "assistant", "content": "辛苦啦~ 要给自己泡杯热茶放松一下吗？🌸"}
            ],
            "temperature": 0.3
        },
        "毒舌": {
            "system_prompt": "你是一个喜欢用犀利吐槽表达关心的朋友...",
            "examples": [
                {"role": "user", "content": "又胖了5斤！"},
                {"role": "assistant", "content": "好家伙！你这是要把体重秤压成分子料理？🏋️"}
            ],
            "temperature": 0.7
        }
    }
    ```

4.  **运行数据生成脚本：**
    确保 `emotion_dialogue_tuner/main.py` 中的文件路径正确指向您的输入问题文件。
    ```bash
    cd emotion-dialogue-finetune/emotion_dialogue_tuner/
    python main.py
    ```
    生成的带风格标签的对话数据将保存到 `outputs/style_chat_data.json` (或在`main.py`中指定的其他路径)。XTuner训练时可能需要将其重命名或配置文件指向此文件，例如 `output.json`。

## 🤖 模型训练步骤 (XTuner)

1.  **准备XTuner配置文件：**
    *   复制一份XTuner官方提供的Qwen1.5的QLoRA配置文件，例如从 `xtuner/configs/qwen/qwen1_5_1_8b_chat/qwen1_5_1_8b_chat_qlora_alpaca_e3.py` 复制到 `xtuner_configs/qwen1_5_1_8b_chat_qlora_alpaca_e3.py`。
    *   **修改配置文件 (`xtuner_configs/qwen1_5_1_8b_chat_qlora_alpaca_e3.py`):**
        ```python
        # PART 1: Settings
        # ...
        pretrained_model_name_or_path = "/path/to/your/Qwen1.5-1.8B-Chat"  # 替换为你的基座模型路径
        # ...
        alpaca_en_path = "/path/to/your/generated_data/output.json"  # 替换为你的微调数据路径
        # ...
        max_length = 512
        batch_size = 4  # 根据显存调整
        max_epochs = 3000 # 教程中发现过拟合，实际可能在6000-7000 iter (非epoch) 效果较好
        # ...
        evaluation_inputs = [ # 用于训练中主观评估的问题
            "男朋友给女主播刷火箭，算精神出轨吗？",
            "喝红酒养生，结果喝到头晕…",
            # ...更多评估问题
        ]
        # ...

        # PART 2: Model & Tokenizer
        # ...
        # 根据教程经验，修改QLoRA参数，例如启用8-bit而非4-bit
        quantization_config=dict(
            type=BitsAndBytesConfig,
            load_in_8bit=True,  # 教程中从False改为True
            # load_in_4bit=False, # 教程中从True改为False
            # llm_int8_threshold=6.0,
            # llm_int8_has_fp16_weight=False,
            # bnb_4bit_compute_dtype=torch.float16,
            # bnb_4bit_use_double_quant=True,
            # bnb_4bit_quant_type='nf4'
            ),
        # ...
        lora_config=dict(
            type=LoraConfig,
            r=32, # 教程中使用32
            lora_alpha=64, # 教程中使用64
            lora_dropout=0.05,
            bias='none',
            task_type='CAUSAL_LM'
        )
        # ...

        # PART 3: Dataset & Dataloader
        # ...
        # 确保数据集类型和路径正确
        dataset=dict(type=load_dataset, path="json", data_files=dict(train=alpaca_en_path)),
        # ...
        ```
        *请仔细参考教程中对配置文件的修改细节，特别是QLoRA参数和数据集加载部分。*

2.  **启动训练：**
    ```bash
    # 进入XTuner配置文件所在目录或提供完整路径
    # 单卡训练
    xtuner train xtuner_configs/qwen1_5_1_8b_chat_qlora_alpaca_e3.py

    # 后台运行并将日志输出到文件
    nohup xtuner train xtuner_configs/qwen1_5_1_8b_chat_qlora_alpaca_e3.py > train_$(date +%Y%m%d_%H%M%S).log 2>&1 &
    ```
    训练过程中注意观察loss变化和`evaluation_inputs`的输出，教程经验表明在6000-7000轮次（iterations）时loss约0.03-0.04，效果较好，避免过拟合。

##🔄 模型转换与合并

训练完成后，LoRA权重保存在 `work_dirs` 下对应配置名称的目录中 (如 `work_dirs/qwen1_5_1_8b_chat_qlora_alpaca_e3/iter_XXXX.pth`)。

1.  **PTH转HuggingFace Adapter格式：**
    ```bash
    # FINETUNE_CFG: 你的XTuner配置文件路径
    # PTH_PATH: 训练得到的.pth权重文件路径
    # SAVE_PATH: 转换后Adapter的保存路径

    xtuner convert pth_to_hf \
        xtuner_configs/qwen1_5_1_8b_chat_qlora_alpaca_e3.py \
        work_dirs/qwen1_5_1_8b_chat_qlora_alpaca_e3/iter_7000.pth \
        work_dirs/qwen1_5_1_8b_chat_qlora_alpaca_e3/iter_7000_hf_adapter
    ```

2.  **Adapter与基座模型合并 (生成完整模型权重)：**
    ```bash
    # LLM_BASE_PATH: 原始Qwen1.5-1.8B-Chat基座模型路径
    # ADAPTER_PATH: 上一步转换得到的Adapter路径
    # SAVE_MERGED_PATH: 合并后完整模型的保存路径

    xtuner convert merge \
        /path/to/your/Qwen1.5-1.8B-Chat \
        work_dirs/qwen1_5_1_8b_chat_qlora_alpaca_e3/iter_7000_hf_adapter \
        work_dirs/qwen1_5_1_8b_chat_qlora_alpaca_e3/Qwen1.5-1.8B-Chat-Emotion-Merged
    ```

## 🚀 模型部署与测试 (LMDeploy & Streamlit)

1.  **准备对话模板 (`chat_template.json`)：**
    XTuner训练时使用的对话模板 (`xtuner/utils/templates.py`中的`qwen_chat`) 可能与LMDeploy不完全兼容。教程中提供了一个`universal_converter.py`脚本（或类似逻辑）将XTuner的模板转换为LMDeploy兼容的JSON格式。
    *   **XTuner `qwen_chat` 模板 (参考):**
        ```python
        qwen_chat=dict(
            SYSTEM=("<|im_start|>system\n{system}<|im_end|>\n"),
            INSTRUCTION=("<|im_start|>user\n{input}<|im_end|>\n" "<|im_start|>assistant\n"),
            SUFFIX="<|im_end|>",
            SUFFIX_AS_EOS=True,
            SEP="\n",
            STOP_WORDS=["<|im_end|>", "<|endoftext|>"],
        )
        ```
    *   **转换后 `outputs/chat_template.json` (示例):**
        ```json
        {
          "meta_instruction": "You are a helpful assistant.",
          "capability": "chat",
          "eosys": "<|im_end|>\n",
          "eoh": "<|im_end|>\n",
          "system": "<|im_start|>system\n{{ system }}<|im_end|>\n",
          "user": "<|im_start|>user\n{{ input }}<|im_end|>",
          "assistant": "<|im_start|>assistant\n",
          "eoa": "<|im_end|>",
          "separator": "\n",
          "stop_words": [
            "<|im_end|>",
            "<|endoftext|>"
          ]
        }
        ```
    *   运行转换脚本 (假设名为 `universal_converter.py` 且已放置在 `emotion_dialogue_tuner/src/`):
        ```bash
        python emotion_dialogue_tuner/src/universal_converter.py
        ```
        这将生成 `chat_template.json` 文件。

2.  **使用LMDeploy启动API服务：**
    ```bash
    # MODEL_PATH: 合并后的完整模型路径
    # CHAT_TEMPLATE_PATH: 上一步生成的chat_template.json路径

    lmdeploy serve api_server \
        work_dirs/qwen1_5_1_8b_chat_qlora_alpaca_e3/Qwen1.5-1.8B-Chat-Emotion-Merged \
        --chat-template outputs/chat_template.json \
        --server-port 23333 # 可选，指定端口
    ```
    *注意：教程中提到自定义对话模板在LMDeploy中加载遇到`'NoneType' object is not subscriptable'`错误，最终通过不加载自定义模板绕过。如果遇到此问题，可尝试不使用 `--chat-template` 参数，LMDeploy会尝试自动推断，但这可能导致非预期的对话格式。请查阅LMDeploy最新文档解决模板兼容性问题。*

3.  **使用OpenAI Python客户端测试API (可选)：**
    创建一个Python脚本 (如 `test_api.py`):
    ```python
    from openai import OpenAI

    client = OpenAI(
        base_url="http://localhost:23333/v1/",  # 确保端口与LMDeploy服务一致
        api_key="your_api_key"  # LMDeploy服务通常不需要真实API Key，任意字符串即可
    )

    chat_history = []
    while True:
        user_input = input("用户: ")
        if user_input.lower() == "exit":
            break
        chat_history.append({"role": "user", "content": user_input})
        try:
            completion = client.chat.completions.create(
                model="Qwen1.5-1.8B-Chat-Emotion-Merged", # 此处model名称与LMDeploy启动时一致
                messages=chat_history,
                temperature=0.7 # 可调整
            )
            response = completion.choices[0].message.content
            print(f"AI: {response}")
            chat_history.append({"role": "assistant", "content": response})
        except Exception as e:
            print(f"Error: {e}")
            break
    ```
    运行: `python test_api.py`

4.  **使用Streamlit进行交互测试 (`emotion_dialogue_tuner/chat_app.py`)：**
    确保 `chat_app.py` 中的 `base_url` 和 `model` 名称正确。
    ```python
    # chat_app.py (部分代码)
    import streamlit as st
    from openai import OpenAI

    # 初始化客户端
    client = OpenAI(
        base_url="http://localhost:23333/v1/", # 确保与LMDeploy服务端口一致
        api_key="your_api_key" # 任意字符串
    )

    st.title("情绪对话模型测试 💬")

    if "messages" not in st.session_state:
        st.session_state.messages = []

    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    if prompt := st.chat_input("请输入您的问题..."):
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)

        with st.chat_message("assistant"):
            message_placeholder = st.empty()
            full_response = ""
            try:
                # 注意：Streamlit应用中，为保持上下文，应传递历史消息
                # 但教程中的chat_app.py示例每次只发送当前消息，这可能导致模型无法进行多轮对话
                # 以下代码修改为传递部分历史（或全部，取决于模型和API承受能力）
                api_messages = [{"role": m["role"], "content": m["content"]} for m in st.session_state.messages]

                response_stream = client.chat.completions.create(
                    model="Qwen1.5-1.8B-Chat-Emotion-Merged", # 确保与LMDeploy启动的模型名一致
                    messages=api_messages, # 发送包含历史的消息
                    stream=True, # 使用流式输出以获得打字机效果
                )
                for chunk in response_stream:
                    if chunk.choices[0].delta.content is not None:
                        full_response += chunk.choices[0].delta.content
                        message_placeholder.markdown(full_response + "▌")
                message_placeholder.markdown(full_response)
            except Exception as e:
                st.error(f"API调用失败: {e}")
                full_response = f"抱歉，出错了: {e}"
            
        st.session_state.messages.append({"role": "assistant", "content": full_response})
    ```
    启动Streamlit应用：
    ```bash
    streamlit run emotion_dialogue_tuner/chat_app.py
    ```
    在浏览器中打开显示的地址进行交互测试。

## 💻 关键代码和配置示例

*   **`config/settings.yaml` (API密钥与路径):**
    ```yaml
    API:
      ZHIPU_API_KEY: "your_zhipu_api_key_here"
      MODEL_NAME: "glm-3-turbo"
    PATHS:
      EMBEDDING_MODEL: "path/to/embedding_model/thomas/text2vec-base-chinese"
    ```

*   **`config/style_config.json` (对话风格定义):**
    ```json
    {
        "温柔": {
            "system_prompt": "你是一个温柔体贴的聊天助手...",
            "examples": [{"role": "user", "content": "..."}, {"role": "assistant", "content": "..."}],
            "temperature": 0.3
        }
        // ...其他风格
    }
    ```

*   **XTuner训练启动命令 (示例):**
    ```bash
    xtuner train xtuner_configs/qwen1_5_1_8b_chat_qlora_alpaca_e3.py
    ```

*   **LMDeploy服务启动命令 (示例):**
    ```bash
    lmdeploy serve api_server work_dirs/.../Qwen1.5-1.8B-Chat-Emotion-Merged --chat-template outputs/chat_template.json
    ```

## 🤔 遇到的问题与解决方案回顾

1.  **`settings.yaml`配置文件找不到：**
    *   **原因：** `config_loader.py`中`root_path`层级配置错误。
    *   **解决：** 修正`Path(__file__).resolve().parent.parent.parent`的层级。
2.  **文件编码错误 (`UnicodeDecodeError: 'gbk'`)：**
    *   **原因：** 文件读取时未指定UTF-8编码。
    *   **解决：** 在所有`open()`函数中添加`encoding="utf-8"`。
3.  **XTuner微调报错 `No module named 'triton.ops'`：**
    *   **原因：** `bitsandbytes`库依赖问题，通常与PyTorch、CUDA版本不兼容。
    *   **解决：** 降级PyTorch版本（如 `torch==2.2.0+cu121`, `torchvision==0.17.0+cu121`），确保CUDA版本匹配（教程中为CUDA 12.1），并重新安装XTuner依赖。
4.  **LMDeploy导入自定义对话模板后报错 (`'NoneType' object is not subscriptable`)：**
    *   **原因：** 自定义的`chat_template.json`格式或内容问题。
    *   **解决/规避：** 教程中未完全解决，测试时通过不加载自定义模板绕过。需仔细检查JSON格式或参考LMDeploy最新文档。
5.  **模型过拟合：**
    *   **原因：** 训练轮数过多，loss过低。
    *   **解决：** 提前停止训练（如6000-7000轮次，loss在0.03-0.04），监控验证集表现。调整XTuner配置以保存更多中间权重。

## 💡 项目总结

本项目成功实践了从数据准备到模型部署的LLM微调全流程，并构建了一个具备特定情绪表达能力的对话模型。主要收获包括：

*   **数据质量是关键：** 高质量、目标明确的数据集对微调效果至关重要。
*   **环境配置挑战：** 依赖包版本（尤其是PyTorch, CUDA, bitsandbytes）的兼容性是常见难点。
*   **迭代与调优：** 模型训练是一个不断迭代和调整参数的过程，需要耐心和细致的观察。
*   **工具链熟悉度：** 掌握XTuner, LMDeploy等工具的使用能极大提高效率。

## 🚀 未来展望

1.  **语音对话集成：** 结合TTS和ASR技术，实现语音交互的情绪对话机器人。
2.  **部署到边缘设备：** 探索将模型量化、剪枝后部署到开发板或嵌入式设备。
3.  **更细致的情绪控制：** 研究更复杂的情绪分类和生成强度控制。
4.  **评估体系完善：** 引入更全面的人工评估和针对情绪表达的客观评估指标。

## 📚 参考资料与链接

*   **智谱AI开放平台:** [https://www.bigmodel.cn/console/overview](https://www.bigmodel.cn/console/overview)
*   **XTuner官方文档:** [https://xtuner.readthedocs.io/zh-cn/latest/](https://xtuner.readthedocs.io/zh-cn/latest/)
*   **OpenCompass官方文档:** [https://doc.opencompass.org.cn/zh_CN/](https://doc.opencompass.org.cn/zh_CN/)
*   **LMDeploy官方文档:** [https://lmdeploy.readthedocs.io/zh-cn/latest/](https://lmdeploy.readthedocs.io/zh-cn/latest/)
*   **QwenLM (Qwen1.5):** [https://github.com/QwenLM/Qwen1.5](https://github.com/QwenLM/Qwen1.5)
*   **Streamlit官方文档:** [https://docs.streamlit.io/](https://docs.streamlit.io/)
*   **数据集参考:**
    *   CDial-GPT: [https://github.com/thu-coai/CDial-GPT](https://github.com/thu-coai/CDial-GPT) (原教程提及，但实际是corpus-dataset/CDial-GPT)
    *   LCCC (魔塔社区): [https://www.modelscope.cn/datasets/OmniData/LCCC](https://www.modelscope.cn/datasets/OmniData/LCCC)
    *   text2vec-base-chinese (嵌入模型): [https://modelscope.cn/models/thomas/text2vec-base-chinese/summary](https://modelscope.cn/models/thomas/text2vec-base-chinese/summary)

---

**免责声明:** 本README根据个人编写的项目操作教程然后通过AI分析整理而成。所有路径、API密钥和具体参数请根据您自己的实际环境和需求进行修改。在执行任何命令前，请确保已备份重要数据。

如有问题或者相应更详细的教程可以查看本人原教程地址：[【项目实战】大模型微调-情绪对话模型](https://blog.ismoyu.cn/xiang-mu-shi-zhan-da-mo-xing-wei-diao-qing-xu-dui-hua-mo-xing/)
