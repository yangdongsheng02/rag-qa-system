# 一 导入库
import os  # 操作系统接口,用于文件操作
from dotenv import load_dotenv

load_dotenv()  # 这会自动从 .env 文件加载环境变量
import warnings  # 导入Python的warnings模块,用于处理警告

warnings.filterwarnings('ignore', category=DeprecationWarning)  # 忽略DeprecationWarning（弃用警告)
os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'  # 国内镜像地址加速国内访问 Hugging Face 模型和数据集

from langchain_text_splitters import \
    RecursiveCharacterTextSplitter  # 文本分割器,用于将长文档分割成小块,便于模型处理和检索,Recursive(递归)意味着它会智能地按照层次结构分割文本
from langchain_community.document_loaders import TextLoader, PyPDFLoader, DirectoryLoader  # 文档加载器:支持TXT,PDF等多种格式
from langchain_huggingface import HuggingFaceEmbeddings  # 嵌入模型:将文本转换为数值向量表示,用于相似性计算
from langchain_chroma import Chroma  # 向量数据库:存储和检索嵌入向量,存储所有文本块的向量,快速查找相似内容
from langchain_core.prompts import PromptTemplate  # 提示词模板,定义如何组织问题,上下文和指令.创建标准化的提示词,提高模型回答质量
from langchain_core.runnables import RunnablePassthrough, \
    RunnableLambda  # LangChain的流程控制组件, RunnablePassthrough:将输入原封不动传递给下一步,在RAG链中传递用户问题
from langchain_core.output_parsers import \
    StrOutputParser  # 输出解析器, StrOutputParser:将模型输出解析为字符串,确保最终输出的是纯文本格式(不然可能是AImessa(内容)的形式)
import requests  # HTTP请求库,调用MiniMax API接口
import json  # json数据处理库,处理API请求和响应的JSON数据
import gradio as gr  # 导入gradio用于构建Web界面。
import re  # 正则表达式


# 二 配置信息类
class Config:
    '''
    集中管理所有配置参数,便于统一管理
    '''
    KNOWLEDGE_BASE_PATH = './knowledge_base'  # 定义知识库文件路径,需要在导入库时指定类型
    PERSIST_DIRECTORY = './chroma_db'  # 向量数据库的保存目录,下次启动时可直接加载,无需重新构建
    EMBED_MODEL_NAME = 'BAAI/bge-small-zh'  # 嵌入模型名称,北京智源研究院的中文小模型,专门为中文优化的嵌入模型
    # MiniMax API配置
    # 从环境变量读取
    MM_API_KEY = os.environ.get('MINIMAX_API_KEY', '')  # 如果没找到环境变量，则返回空字符串
    MM_GROUP_ID = os.environ.get('MINIMAX_GROUP_ID', '')
    MM_API_URL = "https://api.minimaxi.com/v1/text/chatcompletion_v2"  # API地址


# 三 辅助函数(这里用来清洗数据)
def clean_markdown_content(docs):
    """清洗Markdown内容的函数，处理Obsidian内部链接和图片标记"""
    for doc in docs:
        content = doc.page_content

        # 1. 处理Obsidian内部链接 [[目标笔记|别名]] 转换为"别名（目标笔记）"
        # 处理带有别名的链接
        content = re.sub(r'\[\[([^|\]]+)\|([^\]]+)\]\]', r'\2（\1）', content)
        # 处理无别名的链接
        content = re.sub(r'\[\[([^\]]+)\]\]', r'\1', content)

        # 2. 处理图片标记，保留描述文本
        # 将 ![描述](图片地址) 替换为 [图片：描述]
        content = re.sub(r'!\[([^\]]*)\]\([^)]+\)', r'[图片：\1]', content)

        doc.page_content = content
    return docs


# 四 构建知识库函数
def build_knowledge_base():
    '''加载,分割文档,并创建向量数据库'''
    print('开始构建知识库...')

    # 1.检查文件是否存在
    dir_path = Config.KNOWLEDGE_BASE_PATH  # 获取配置中的文件路径

    if not os.path.exists(dir_path):
        # 如果目录不存在，创建空目录
        print(f"知识库目录不存在，创建目录: {dir_path}")
        os.makedirs(dir_path, exist_ok=True)
        print(f"已创建知识库目录: {dir_path}")
        print(f"请将知识库复制到 {dir_path} 目录中，然后重新运行程序。")
        # 返回None，表示没有构建知识库
        return None

    print(f"从目录加载知识库: {dir_path}")

    # 2. 使用DirectoryLoader加载所有.md文件
    loader = DirectoryLoader(
        path=dir_path,
        glob="**/*.md",  # 匹配所有.md文件
        loader_cls=TextLoader,
        loader_kwargs={'encoding': 'utf-8'},  # 使用UTF-8编码
        exclude=["**/.obsidian/**", "**/附件/**", "**/assets/**"],  # 排除特定目录
        show_progress=True,  # 显示加载进度
        use_multithreading=True  # 使用多线程加速
    )

    # 3.加载文档
    documents = loader.load()  # loader.load():执行文档加载,返回一个文档对象列表,PDF文档每页为一个Document对象，TXT文档整个文件为一个Document对象
    print(f"已加载文档，共 {len(documents)} 个.md文件")
    if len(documents) == 0:
        print("知识库目录中没有Markdown文件")
        print(f"请将笔记复制到 {dir_path} 目录中")
        return None

    # 4. 清洗Markdown内容（处理内部链接、图片等）
    documents = clean_markdown_content(documents)
    print("已完成Markdown内容清洗")

    # 5.创建文本分割器,分割文本为小块,便于模型处理(LLM有输入长度限制)
    text_splitter = RecursiveCharacterTextSplitter(  # recursive递归,按层智能切割
        chunk_size=500,  # 每个文本块最多500字符
        chunk_overlap=50,  # 块之间的重叠字符,保持上下文,相邻块重叠50字符，防止信息割裂
        separators=["\n\n", "\n# ", "\n## ", "\n### ", "\n", "。", "，", " ", ""]  # 优先按段落和标题分割
    )
    splits = text_splitter.split_documents(documents)  # 执行分割,返回更小的Document对象列表
    print(f"文档已分割为 {len(splits)} 个文本块")

    # 6.创建嵌入模型(将文本转换为向量),
    # 加载预训练的中文嵌入模型
    embeddings = HuggingFaceEmbeddings(
        model_name=Config.EMBED_MODEL_NAME,
        model_kwargs={'device': 'cpu'},  # 使用CPU，GPU可改为 'cuda'
        encode_kwargs={'normalize_embeddings': True}
        # normalize_embeddings=True：归一化嵌入向量使所有向量长度为1，便于余弦相似度计算. 大概意思就是把那些影响较大的因素的影响变小
    )

    # 7.创建向量数据库
    try:
        vectordb = Chroma.from_documents(
            documents=splits,  # 输入分割后的文本块
            embedding=embeddings,  # 使用指定的嵌入模型
            persist_directory=Config.PERSIST_DIRECTORY  # 将当前内存中的向量数据库（包括索引、向量数据、元数据等）保存到指定目录
        )
        print(f"向量数据库已创建并保存至：{Config.PERSIST_DIRECTORY}")
    except Exception as e:
        print(f"创建向量数据库时出错: {str(e)}")
        # 尝试不带 persist_directory 创建
        vectordb = Chroma.from_documents(
            documents=splits,
            embedding=embeddings
        )
        print("向量数据库未持久化，重启后需重新构建")

    print(f"知识库构建完成！包含 {len(documents)} 个文档，{len(splits)} 个文本块")
    return vectordb  # 返回向量数据库对象供后续使用


# 五 创建检索器函数
def create_retrieve(vectordb):
    """创建检索器，负责从向量库中找出与问题相关的文本块"""
    # 搜索最相关的6个文本块
    retriever = vectordb.as_retriever(
        search_kwargs={'k': 6}  # k=x:f返回最相似的x个文本块,需个块平衡回答质量与处理时间(这个回答会很慢,块太少回答质量很差)
    )
    return retriever


# 六,MiniMax LLM封装类
class MiniMaxLLM:
    """封装MiniMax API调用"""

    @staticmethod  # 静态方法,不用创建类实例即可调用
    def invoke(prompt: str) -> str:  # 类型提示：输入str，返回str
        """调用MiniMax API生成回复"""
        # 处理不同格式的提示词输入
        if hasattr(prompt, 'to_string'):  # hasattr 函数用于检查对象是否具有指定的属性或方法。它接受两个参数：对象和属性名，并返回一个布尔值。
            # 检查prompt是否有to_string方法（可能是PromptTemplate对象）
            prompt_content = prompt.to_string()
        elif not isinstance(prompt, str):
            # 如果不是字符串类型，转换为字符串
            prompt_content = str(prompt)
        else:
            prompt_content = prompt  # 已经是字符串，直接使用

        # 通常在使用API时，需要设置请求头（headers），以提供必要的认证信息和指定请求体的格式
        headers = {
            'Authorization': f'Bearer {Config.MM_API_KEY}',  # Authorization: HTTP标准认证头,Bearer令牌认证: 一种认证方案（类似"钥匙"
            'Content-Type': 'application/json',  # Content-Type: 告诉服务器请求体的格式,application/json: 表示数据是JSON格式
            "Group-Id": Config.MM_GROUP_ID,
        }
        # 请求体配置
        payload = {
            'model': "abab5.5-chat",  # 指定模型版本
            'messages': [
                {
                    'role': 'system',
                    'content': "你是一个专业的助手，严格根据提供的资料回答问题。如果资料中没有相关信息，请直接说明'根据资料无法回答此问题'，不要编造信息。"
                },
                {
                    "role": "user",
                    "content": prompt_content
                }
            ],
            "temperature": 0.1,  # 温度参数：控制随机性，0.1表示低随机性,保证回答稳定性
            "top_p": 0.7,  # 核采样参数：限制词汇选择范围
            "stream": False,  # 非流式响应：一次性返回完整答案
            "max_tokens": 1024,  # 最大生成长度：1024个token
        }

        try:  # 异常处理：捕获可能的网络或API错误
            # 发送POST请求
            response = requests.post(
                Config.MM_API_URL,
                headers=headers,
                json=payload,  # 自动序列化为JSON
                timeout=30  # 30秒超时
            )

            # 检查HTTP状态码
            if response.status_code != 200:
                # API调用失败
                error_msg = f" API请求失败，状态码: {response.status_code}"
                try:
                    # 尝试解析错误详情
                    error_detail = response.json()
                    error_msg += f"\n错误详情: {json.dumps(error_detail, ensure_ascii=False)}"
                    # json.dumps：将Python对象转换为JSON字符串
                    # ensure_ascii=False：允许非ASCII字符（如中文）
                except:
                    # 如果响应不是JSON格式，直接显示文本
                    error_msg += f"\n响应内容: {response.text[:500]}"
                    # 只显示前500字符，避免过长输出
                return error_msg

            # 解析成功响应
            result = response.json()  # 将JSON响应转换为Python字典

            # 提取回答内容（适配API可能的不同响应格式）
            if "choices" in result and result["choices"]:
                choice = result["choices"][0]  # 获取第一个选择
                if "message" in choice and "content" in choice["message"]:
                    return choice["message"]["content"]

            if "reply" in result:  # 另一种可能的响应格式
                return result["reply"]

            # 无法提取答案的情况
            return f"无法从API响应中提取答案"

        except requests.exceptions.Timeout:
            # 网络超时异常
            return "API请求超时"
        except Exception as e:
            # 其他所有异常
            return f"调用API时出错: {str(e)}"


# 七、Agent类（新添加的部分）####################################################
class SimpleToolAgent:
    """简单的工具调用Agent，完全独立于RAG系统"""

    def __init__(self, rag_chain):
        self.rag_chain = rag_chain  # 保留但不使用

    def call_api_directly(self, prompt, system_prompt=None):
        """直接调用MiniMax API，完全绕过RAG的限制"""
        headers = {
            'Authorization': f'Bearer {Config.MM_API_KEY}',
            'Content-Type': 'application/json',
            "Group-Id": Config.MM_GROUP_ID,
        }

        # 使用无限制的系统提示词
        if system_prompt is None:
            system_prompt = "你是一个智能助手，请准确回答用户的问题。如果是计算问题请给出详细步骤，如果是概念解释请清晰说明。"

        payload = {
            'model': "abab5.5-chat",
            'messages': [
                {
                    'role': 'system',
                    'content': system_prompt
                },
                {
                    "role": "user",
                    "content": prompt
                }
            ],
            "temperature": 0.1,
            "top_p": 0.7,
            "stream": False,
            "max_tokens": 1024,
        }

        try:
            response = requests.post(
                Config.MM_API_URL,
                headers=headers,
                json=payload,
                timeout=30
            )

            if response.status_code == 200:
                result = response.json()
                if "choices" in result and result["choices"]:
                    return result["choices"][0]["message"]["content"]
                elif "reply" in result:
                    return result["reply"]
                else:
                    return "已处理您的请求"
            else:
                return f"请求失败，状态码: {response.status_code}"
        except Exception as e:
            return f"处理请求时出错: {str(e)}"

    def detect_task_type(self, question):
        """检测问题类型"""
        question_lower = question.lower()

        # 检查是否是多步骤任务（必须同时有"先"和"然后"）
        if "先" in question_lower and ("然后" in question_lower or "接着" in question_lower):
            return "multi_step"
        elif any(word in question_lower for word in
                 ["计算", "等于", "多少", "+", "-", "*", "/", "平方", "加", "减", "乘", "除"]):
            return "calculation"
        elif any(word in question_lower for word in ["解释", "什么是", "定义", "概念", "说明"]):
            return "explanation"
        else:
            return "general"

    def split_multi_step(self, question):
        """分割多步骤问题"""
        # 找到分割点
        if "然后" in question:
            split_word = "然后"
        elif "接着" in question:
            split_word = "接着"
        else:
            return [question]

        parts = question.split(split_word)
        steps = []

        for part in parts:
            cleaned = part.strip()
            # 去掉"先"字
            if cleaned.startswith("先"):
                cleaned = cleaned[1:].strip()
            # 去掉开头的中文标点
            cleaned = cleaned.lstrip('，。：:')
            if cleaned:
                steps.append(cleaned)

        return steps

    def process_single_step(self, question, step_type):
        """处理单步骤问题"""
        if step_type == "calculation":
            # 计算问题
            prompt = f"请计算这个问题：{question}。请给出详细的计算步骤和最终结果。"
            system_prompt = "你是一个数学助手，请准确计算用户的问题并给出详细步骤。"
            result = self.call_api_directly(prompt, system_prompt)
            return {
                "type": "计算",
                "result": result
            }
        elif step_type == "explanation":
            # 解释问题
            prompt = f"请详细解释：{question}。包括：1.定义 2.原理 3.应用场景 4.相关技术"
            system_prompt = "你是一个技术专家，请清晰准确地解释技术概念。"
            result = self.call_api_directly(prompt, system_prompt)
            return {
                "type": "解释",
                "result": result
            }
        else:
            # 一般问题
            result = self.call_api_directly(question)
            return {
                "type": "通用",
                "result": result
            }

    def run(self, question):
        """主运行方法"""
        task_type = self.detect_task_type(question)

        if task_type == "multi_step":
            # 处理多步骤任务
            steps = self.split_multi_step(question)

            if len(steps) <= 1:
                # 如果不是真正的多步骤，按单步骤处理
                step_type = self.detect_task_type(question)
                result = self.process_single_step(question, step_type)

                response = f"""
🤖 **Agent工作流程**

**问题分析**: {question} → {result['type']}任务

**执行结果**:
{result['result']}

---
*Agent演示：展示了工具选择和执行过程*
"""
                return response

            # 处理每个步骤
            step_results = []
            for i, step in enumerate(steps):
                step_type = self.detect_task_type(step)
                result = self.process_single_step(step, step_type)
                step_results.append({
                    "index": i + 1,
                    "step": step,
                    "type": result["type"],
                    "result": result["result"]
                })

            # 构建多步骤响应
            step_summary = []
            step_details = []

            for sr in step_results:
                step_summary.append(f"{sr['index']}. {sr['step']} → {sr['type']}工具")
                step_details.append(f"**步骤{sr['index']}** ({sr['type']}工具):\n{sr['result']}")

            response = f"""
🤖 **多步骤任务执行报告**

**原始问题**: {question}

**任务分解**:
{chr(10).join(step_summary)}

**执行结果**:
{chr(10).join(step_details)}

---
*Agent演示：展示了多步骤任务的处理能力和智能工具选择*
"""
            return response
        else:
            # 处理单步骤任务
            result = self.process_single_step(question, task_type)

            response = f"""
🤖 **Agent工作流程**

**问题分析**: {question} → {result['type']}任务

**执行结果**:
{result['result']}

---
*Agent演示：展示了工具选择和执行过程*
"""
            return response
# 八 构建RAG应用链
def create_rag_chain(retriever):
    # 定义提示词模板
    template = """请严格依据以下提供的背景资料来回答问题。如果资料中没有相关信息，请直接说明"根据资料无法回答此问题"，不要编造信息。

    **特别指令**：如果用户的问题是要求总结、概述或寻找主题，请你仔细分析所有提供的资料，进行归纳、分类和概括，梳理出清晰的结构。

    背景资料：
    {context}  # 占位符：将被检索到的文档替换

    问题：{question}  # 占位符：将被用户问题替换

    请基于资料提供准确、详细的回答："""

    # 创建promptTemplate对象
    prompt = PromptTemplate.from_template(template)  # 从template模板字符串创建提示词模板

    # 定义文档格式化函数
    def format_docs(docs):
        # 将Document对象列表连接为单个字符串
        return '\n\n'.join([doc.page_content for doc in docs])  ## 列表推导式：提取每个Document的page_content

    # 使用LangChain表达式语言(LCEL)构建处理链,检索 -> 格式化 -> 生成
    rag_chain = (
        # 1.准备输入数据字典
            {'context': retriever | format_docs, 'question': RunnablePassthrough()}  # retriever|format_docs先检索再格式化
            | prompt  # 填充提示词模板
            | MiniMaxLLM.invoke  # 调用LLM根据提示词生成回答
            | StrOutputParser()
    )
    return rag_chain


# 九 Gradio界面类(Web前端构建)
class ChatInterface:
    def __init__(self, rag_chain):
        self.rag_chain = rag_chain  # 保存RAG链
        self.chat_history = []  # 存储对话历史
        self.agent = SimpleToolAgent(rag_chain)  # 新加：创建Agent实例
        self.mode = "rag"  # 新加：模式标记，rag 或 agent

    def change_mode(self, new_mode):
        """切换模式"""
        self.mode = new_mode
        if new_mode == "agent":
            return "🤖 已切换到Agent模式！现在我会展示工具选择和决策过程。"
        else:
            return "🔍 已切换到RAG模式（纯检索增强生成）。"

    def add_message(self, role, content):
        """添加消息到聊天历史"""
        self.chat_history.append({"role": role, "content": content})

    def respond(self, message, chat_history):
        """处理用户消息并返回响应"""
        # 检查是否是模式切换命令
        if message.lower() in ["/agent", "/rag", "/mode agent", "/mode rag"]:
            if "agent" in message.lower():
                reply = self.change_mode("agent")
            else:
                reply = self.change_mode("rag")
            self.add_message("assistant", reply)
            return self.chat_history

        # 添加用户消息到历史
        self.add_message("user", message)

        try:
            # 显示思考状态
            thinking_msg = "正在思考..."
            self.add_message("assistant", thinking_msg)

            # 根据模式选择处理方式
            if self.mode == "agent":
                # 使用Agent处理
                answer = self.agent.run(message)
            else:
                # 使用纯RAG处理
                answer = self.rag_chain.invoke(message)

            # 更新最后一条消息为实际回答
            self.chat_history[-1] = {"role": "assistant", "content": answer}

            # 返回更新后的聊天历史
            return self.chat_history
        except Exception as e:
            error_msg = f"系统错误: {str(e)}"
            self.add_message("assistant", error_msg)
            return self.chat_history

    def clear_chat(self):
        """清空聊天历史"""
        self.chat_history = []
        self.mode = "rag"  # 清空时重置模式
        return []

    def create_interface(self):
        """创建Gradio界面"""
        with gr.Blocks(title="智能知识库问答系统 - RAG + Agent演示") as interface:

            # 标题区域
            gr.Markdown("# 智能知识库问答系统")
            gr.Markdown("**RAG + Agent 演示系统** - 展示检索增强生成与Agent工具调用")

            with gr.Row():
                with gr.Column(scale=3):
                    # 聊天机器人组件
                    chatbot = gr.Chatbot(
                        height=500,
                        label="对话记录",
                        value=self.chat_history
                    )

                    # 输入区域
                    with gr.Row():
                        msg = gr.Textbox(
                            placeholder="请输入您的问题... 输入 /agent 切换Agent模式，/rag 切换RAG模式",
                            show_label=False,
                            scale=4,
                            container=False,
                            lines=2
                        )
                        submit_btn = gr.Button("发送", variant="primary", scale=1)

                    # 功能按钮行
                    with gr.Row():
                        clear_btn = gr.Button("清空对话", variant="secondary")

                with gr.Column(scale=1):
                    # 系统信息面板（新加模式显示）
                    gr.Markdown("### 系统信息")

                    # 当前模式显示
                    mode_display = gr.Markdown(
                        f"**当前模式**: {'🤖 Agent模式' if self.mode == 'agent' else '🔍 RAG模式'}")

                    # 模式切换按钮（新加）
                    gr.Markdown("### 模式切换")
                    with gr.Row():
                        agent_btn = gr.Button("切换到🤖 Agent模式", variant="primary")
                        rag_btn = gr.Button("切换到🔍 RAG模式", variant="secondary")

                    # Agent功能介绍（新加）
                    gr.Markdown("""
                    **🤖 Agent模式功能：**
                    1. 自动选择工具（搜索/计算/解释）
                    2. 展示决策过程
                    3. 多工具协作演示

                    **可用命令：**
                    - `/agent` 或 `/mode agent`：切换到Agent模式
                    - `/rag` 或 `/mode rag`：切换到RAG模式
                    """)

                    # 示例问题区域（增加Agent示例）
                    gr.Markdown("### 试试这些问题：")

                    # 普通示例问题
                    examples = [
                        "神经网络的概念",
                        "为什么要分词和编码",
                        "概括下注意力头数的作用",
                    ]

                    # Agent模式专用示例（新加）
                    agent_examples = [
                        "计算一下(15 + 27) * 3等于多少",
                        "解释一下Transformer的概念",
                        "先计算3的平方，然后解释一下什么是注意力机制",
                        "什么是梯度下降？计算一下10的平方根",
                    ]

                    gr.Markdown("**普通问题：**")
                    for example in examples:
                        btn = gr.Button(
                            example[:25] + "..." if len(example) > 25 else example,
                            size="sm",
                            variant="secondary"
                        )
                        btn.click(lambda q=example: q, None, msg)

                    gr.Markdown("**Agent演示问题：**")
                    for example in agent_examples:
                        btn = gr.Button(
                            example[:25] + "..." if len(example) > 25 else example,
                            size="sm",
                            variant="primary"
                        )
                        btn.click(lambda q=example: q, None, msg)

            # 事件绑定
            # 发送按钮
            submit_btn.click(
                fn=self.respond,
                inputs=[msg, chatbot],
                outputs=[chatbot]
            ).then(
                lambda: "", None, msg
            )

            # 回车发送
            msg.submit(
                fn=self.respond,
                inputs=[msg, chatbot],
                outputs=[chatbot]
            ).then(
                lambda: "", None, msg
            )

            # 清空对话
            clear_btn.click(
                fn=self.clear_chat,
                inputs=None,
                outputs=[chatbot]
            )

            # 模式切换按钮事件（新加）
            def switch_to_agent():
                return "/agent"

            def switch_to_rag():
                return "/rag"

            agent_btn.click(
                fn=switch_to_agent,
                inputs=None,
                outputs=msg
            )

            rag_btn.click(
                fn=switch_to_rag,
                inputs=None,
                outputs=msg
            )

            # 更新模式显示的响应函数（新加）
            agent_btn.click(
                fn=lambda: "**当前模式**: 🤖 Agent模式",
                inputs=None,
                outputs=mode_display
            )

            rag_btn.click(
                fn=lambda: "**当前模式**: 🔍 RAG模式",
                inputs=None,
                outputs=mode_display
            )

        return interface


# 十 主函数(程序入口)
def main():
    # 标题
    print('=' * 50)
    print('个人知识库问答系统(RAG) - 包含Agent演示')
    print('=' * 50)

    # 显示知识库信息
    print(f"知识库目录: {Config.KNOWLEDGE_BASE_PATH}")

    if os.path.exists(Config.KNOWLEDGE_BASE_PATH):
        md_files = []
        for root, dirs, files in os.walk(Config.KNOWLEDGE_BASE_PATH):
            for file in files:
                if file.endswith('.md'):
                    md_files.append(os.path.join(root, file))

        print(f"发现 {len(md_files)} 个Markdown文件")
        if len(md_files) > 0:
            print("您的笔记文件:")
            for i, file in enumerate(md_files[:8]):
                print(f"  {i + 1}. {os.path.basename(file)}")
            if len(md_files) > 8:
                print(f"  ... 还有 {len(md_files) - 8} 个文件")
        else:
            print(" 知识库目录中没有Markdown文件")
            print(f"请将知识库文件复制到 {Config.KNOWLEDGE_BASE_PATH} 目录中")
    else:
        print(f"知识库目录不存在")

    # 初始化知识库
    if not os.path.exists(Config.PERSIST_DIRECTORY):
        # 检查向量数据库是否存在
        print("未找到已构建的知识库，开始初始化...")
        vectordb = build_knowledge_base()  # 调用函数构建向量知识库
    else:
        print('加载已有知识库')
        embeddings = HuggingFaceEmbeddings(model_name=Config.EMBED_MODEL_NAME)
        vectordb = Chroma(
            persist_directory=Config.PERSIST_DIRECTORY,  # 向量数据存储位置
            embedding_function=embeddings  # 使用指定模型进行向量化
        )

    # 创建RAG链
    retriever = create_retrieve(vectordb)
    rag_chain = create_rag_chain(retriever)

    print('=' * 50)
    print('🤖 Agent功能已启用！')
    print('可用命令：')
    print('  - /agent 或 /mode agent：切换到Agent模式')
    print('  - /rag 或 /mode rag：切换到RAG模式')
    print('=' * 50)
    print('系统准备就绪')
    print('正在启动Web界面...')

    # 创建并启动Gradio界面
    chat_interface = ChatInterface(rag_chain)
    interface = chat_interface.create_interface()

    print("请在浏览器中访问：http://localhost:7860")
    print("界面中有专门的Agent演示区域和示例问题")

    # 启动Gradio服务器
    interface.launch(
        server_name="0.0.0.0",
        server_port=7860,
        share=False,
        show_error=True,
        quiet=False,
        theme=gr.themes.Soft()
    )


# 程序入口点
if __name__ == "__main__":
    main()