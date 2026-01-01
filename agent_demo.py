# 一 导入库
import os
from dotenv import load_dotenv

load_dotenv()
import warnings

warnings.filterwarnings('ignore', category=DeprecationWarning)
os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'

from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import TextLoader, DirectoryLoader
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
import requests
import json
import gradio as gr
import re
import math
import random


# 二 配置信息类
class Config:
    KNOWLEDGE_BASE_PATH = './knowledge_base'
    PERSIST_DIRECTORY = './chroma_db'
    EMBED_MODEL_NAME = 'BAAI/bge-small-zh'
    MM_API_KEY = os.environ.get('MINIMAX_API_KEY', '')
    MM_GROUP_ID = os.environ.get('MINIMAX_GROUP_ID', '')
    MM_API_URL = "https://api.minimaxi.com/v1/text/chatcompletion_v2"


# 三 辅助函数
def clean_markdown_content(docs):
    for doc in docs:
        content = doc.page_content
        content = re.sub(r'\[\[([^|\]]+)\|([^\]]+)\]\]', r'\2（\1）', content)
        content = re.sub(r'\[\[([^\]]+)\]\]', r'\1', content)
        content = re.sub(r'!\[([^\]]*)\]\([^)]+\)', r'[图片：\1]', content)
        doc.page_content = content
    return docs


# 四 构建知识库函数
def build_knowledge_base():
    print('开始构建知识库...')

    dir_path = Config.KNOWLEDGE_BASE_PATH

    if not os.path.exists(dir_path):
        print(f"知识库目录不存在，创建目录: {dir_path}")
        os.makedirs(dir_path, exist_ok=True)
        return None

    loader = DirectoryLoader(
        path=dir_path,
        glob="**/*.md",
        loader_cls=TextLoader,
        loader_kwargs={'encoding': 'utf-8'},
        exclude=["**/.obsidian/**", "**/附件/**", "**/assets/**"],
        show_progress=True,
        use_multithreading=True
    )

    documents = loader.load()
    if len(documents) == 0:
        print("知识库目录中没有Markdown文件")
        return None

    documents = clean_markdown_content(documents)

    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,
        chunk_overlap=50,
        separators=["\n\n", "\n# ", "\n## ", "\n### ", "\n", "。", "，", " ", ""]
    )
    splits = text_splitter.split_documents(documents)

    embeddings = HuggingFaceEmbeddings(
        model_name=Config.EMBED_MODEL_NAME,
        model_kwargs={'device': 'cpu'},
        encode_kwargs={'normalize_embeddings': True}
    )

    try:
        vectordb = Chroma.from_documents(
            documents=splits,
            embedding=embeddings,
            persist_directory=Config.PERSIST_DIRECTORY
        )
        print(f"向量数据库已保存至：{Config.PERSIST_DIRECTORY}")
    except Exception as e:
        print(f"创建向量数据库时出错: {str(e)}")
        vectordb = Chroma.from_documents(
            documents=splits,
            embedding=embeddings
        )
        print("向量数据库未持久化")

    print(f"知识库构建完成！包含 {len(documents)} 个文档，{len(splits)} 个文本块")
    return vectordb


# 五 创建检索器函数
def create_retrieve(vectordb):
    retriever = vectordb.as_retriever(
        search_kwargs={'k': 6}
    )
    return retriever


# 六,MiniMax LLM封装类
class MiniMaxLLM:
    @staticmethod
    def invoke(prompt: str) -> str:
        if hasattr(prompt, 'to_string'):
            prompt_content = prompt.to_string()
        elif not isinstance(prompt, str):
            prompt_content = str(prompt)
        else:
            prompt_content = prompt

        headers = {
            'Authorization': f'Bearer {Config.MM_API_KEY}',
            'Content-Type': 'application/json',
            "Group-Id": Config.MM_GROUP_ID,
        }

        payload = {
            'model': "abab5.5-chat",
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

            if response.status_code != 200:
                error_msg = f" API请求失败，状态码: {response.status_code}"
                try:
                    error_detail = response.json()
                    error_msg += f"\n错误详情: {json.dumps(error_detail, ensure_ascii=False)}"
                except:
                    error_msg += f"\n响应内容: {response.text[:500]}"
                return error_msg

            result = response.json()

            if "choices" in result and result["choices"]:
                choice = result["choices"][0]
                if "message" in choice and "content" in choice["message"]:
                    return choice["message"]["content"]

            if "reply" in result:
                return result["reply"]

            return "无法从API响应中提取答案"

        except requests.exceptions.Timeout:
            return "API请求超时"
        except Exception as e:
            return f"调用API时出错: {str(e)}"


# 七 工具函数
class ToolManager:
    """工具管理器"""

    @staticmethod
    def calculate_tool(expression: str) -> str:
        """计算器工具"""
        try:
            # 移除可能的中文说明，提取纯数学表达式
            expr = expression.lower()
            # 替换中文运算符
            expr = expr.replace('加', '+').replace('减', '-').replace('乘', '*').replace('除以', '/').replace('除', '/')

            # 只允许安全字符
            safe_chars = set('0123456789+-*/.() ')
            expr_clean = ''.join(c for c in expr if c in safe_chars)

            if not expr_clean:
                return "未识别到有效的数学表达式"

            result = eval(expr_clean)
            return f"{result}"
        except Exception as e:
            return f"计算失败: {str(e)}"

    @staticmethod
    def summarize_tool(text: str) -> str:
        """总结工具"""
        prompt = f"请将以下内容用中文总结成3-5个要点：\n\n{text[:1000]}"
        return MiniMaxLLM.invoke(prompt)

    @staticmethod
    def translate_tool(text: str, target_lang: str = "英文") -> str:
        """翻译工具"""
        prompt = f"将以下内容翻译成{target_lang}：\n\n{text[:500]}"
        return MiniMaxLLM.invoke(prompt)


# 八 智能Agent类（支持多步骤任务）
class IntelligentAgent:
    """智能Agent，能够处理多步骤任务"""

    def __init__(self, rag_chain):
        self.rag_chain = rag_chain
        self.tools = ToolManager()
        self.task_planner = TaskPlanner()

    def process_multistep_task(self, query: str) -> str:
        """处理多步骤任务"""
        # 尝试识别多步骤任务
        steps = self.task_planner.identify_steps(query)

        if len(steps) <= 1:
            # 单步任务，使用原来的逻辑
            return self.process_single_task(query)

        # 多步骤任务
        results = []
        for i, step in enumerate(steps, 1):
            step_result = self.process_single_step(step, i)
            results.append(step_result)

        # 组合所有结果
        final_result = self.combine_results(results, query)
        return final_result

    def process_single_step(self, step: str, step_num: int) -> dict:
        """处理单个步骤"""
        step_type = self.task_planner.classify_step(step)

        if step_type == "calculation":
            result = self.tools.calculate_tool(step)
            return {"type": "calculation", "step": step, "result": result, "step_num": step_num}
        elif step_type == "rag_query":
            result = self.rag_chain.invoke(step)
            return {"type": "rag_query", "step": step, "result": result, "step_num": step_num}
        elif step_type == "summary":
            result = self.tools.summarize_tool(step)
            return {"type": "summary", "step": step, "result": result, "step_num": step_num}
        else:
            # 默认使用RAG
            result = self.rag_chain.invoke(step)
            return {"type": "rag_query", "step": step, "result": result, "step_num": step_num}

    def process_single_task(self, query: str) -> str:
        """处理单步任务"""
        step_type = self.task_planner.classify_step(query)

        if step_type == "calculation":
            result = self.tools.calculate_tool(query)
            return f" **计算结果**\n\n{query} = {result}"
        elif step_type == "rag_query":
            result = self.rag_chain.invoke(query)
            return f" **知识库查询结果**\n\n{result}"
        elif step_type == "summary":
            result = self.tools.summarize_tool(query)
            return f" **总结结果**\n\n{result}"
        else:
            result = self.rag_chain.invoke(query)
            return f" **知识库查询结果**\n\n{result}"

    def combine_results(self, results: list, original_query: str) -> str:
        """组合多步骤结果"""
        final_output = " **多步骤任务执行完成**\n\n"
        final_output += f"**原始指令**: {original_query}\n\n"
        final_output += "**执行过程**:\n\n"

        for result in results:
            if result["type"] == "calculation":
                final_output += f"{result['step_num']}.  **计算**: {result['step']}\n"
                final_output += f"   结果: {result['result']}\n\n"
            elif result["type"] == "rag_query":
                final_output += f"{result['step_num']}.  **知识查询**: {result['step']}\n"
                final_output += f"   结果: {result['result']}\n\n"
            elif result["type"] == "summary":
                final_output += f"{result['step_num']}.  **总结**: {result['step']}\n"
                final_output += f"   结果: {result['result']}\n\n"

        final_output += "---\n*多步骤任务执行完成*"
        return final_output

    def execute(self, query: str) -> str:
        """执行Agent决策"""
        # 检查是否是多步骤任务
        if self.task_planner.is_multistep_task(query):
            return self.process_multistep_task(query)
        else:
            return self.process_single_task(query)


# 九 任务规划器类
class TaskPlanner:
    """任务规划器，识别和分解多步骤任务"""

    def __init__(self):
        self.multistep_keywords = [
            "先", "然后", "接着", "再", "之后", "最后",
            "第一步", "第二步", "第三步", "第四步", "第五步",
            "首先", "其次", "再次", "最后"
        ]

        self.calculation_keywords = [
            "计算", "等于", "多少", "加", "减", "乘", "除",
            "+", "-", "*", "/", "(", ")", "平方", "立方", "根号"
        ]

        self.summary_keywords = [
            "总结", "概括", "概述", "要点", "主要", "核心"
        ]

    def is_multistep_task(self, query: str) -> bool:
        """判断是否是多步骤任务"""
        query_lower = query.lower()

        # 检查是否包含多个任务关键词
        keyword_count = 0
        for keyword in self.multistep_keywords:
            if keyword in query_lower:
                keyword_count += 1
                if keyword_count >= 2:  # 至少有两个任务指示词
                    return True

        # 检查是否有明确的步骤指示
        if "第一步" in query_lower and ("第二步" in query_lower or "然后" in query_lower):
            return True

        # 检查是否有"先...再..."模式
        if "先" in query_lower and ("再" in query_lower or "然后" in query_lower):
            return True

        return False

    def identify_steps(self, query: str) -> list:
        """识别任务步骤"""
        steps = []
        query_lower = query.lower()

        # 尝试按常见模式分割
        if "先" in query_lower and "再" in query_lower:
            # 处理"先...再..."模式
            parts = query.split("先")[1].split("再")
            if len(parts) >= 2:
                steps.append(f"先{parts[0]}".strip())
                steps.append(f"再{parts[1]}".strip())
        elif "然后" in query_lower:
            # 处理"...然后..."模式
            parts = query.split("然后")
            steps = [p.strip() for p in parts if p.strip()]
        elif "接着" in query_lower:
            # 处理"...接着..."模式
            parts = query.split("接着")
            steps = [p.strip() for p in parts if p.strip()]
        elif "第一步" in query_lower:
            # 处理"第一步...第二步..."模式
            import re
            step_pattern = r'第[一二三四五]步[:：]?\s*(.*?)(?=第[一二三四五]步|$)'
            matches = re.findall(step_pattern, query)
            if matches:
                steps = [m.strip() for m in matches if m.strip()]

        # 如果以上模式都没匹配到，尝试按标点分割
        if not steps:
            # 按句号、分号、逗号等分割
            import re
            sentences = re.split(r'[。；;，,]', query)
            steps = [s.strip() for s in sentences if s.strip() and len(s.strip()) > 3]

        # 清理步骤文本
        cleaned_steps = []
        for step in steps:
            # 移除步骤指示词
            step_clean = step
            for keyword in self.multistep_keywords:
                if step_clean.startswith(keyword):
                    step_clean = step_clean[len(keyword):].strip()
            cleaned_steps.append(step_clean)

        return cleaned_steps if cleaned_steps else [query]

    def classify_step(self, step: str) -> str:
        """分类单个步骤的类型"""
        step_lower = step.lower()

        # 检查是否是计算任务
        if any(keyword in step_lower for keyword in self.calculation_keywords):
            # 进一步确认是否是真正的计算（包含数字）
            if any(char.isdigit() for char in step):
                return "calculation"

        # 检查是否是总结任务
        if any(keyword in step_lower for keyword in self.summary_keywords):
            return "summary"

        # 默认是RAG查询
        return "rag_query"


# 十 RAG链函数
def create_rag_chain(retriever):
    template = """请严格依据以下提供的背景资料来回答问题。如果资料中没有相关信息，请直接说明"根据资料无法回答此问题"，不要编造信息。

    **特别指令**：如果用户的问题是要求总结、概述或寻找主题，请你仔细分析所有提供的资料，进行归纳、分类和概括，梳理出清晰的结构。

    背景资料：
    {context}

    问题：{question}

    请基于资料提供准确、详细的回答："""

    prompt = PromptTemplate.from_template(template)

    def format_docs(docs):
        return '\n\n'.join([doc.page_content for doc in docs])

    rag_chain = (
            {'context': retriever | format_docs, 'question': RunnablePassthrough()}
            | prompt
            | MiniMaxLLM.invoke
            | StrOutputParser()
    )
    return rag_chain


# 十一 Gradio界面类
class ChatInterface:
    def __init__(self, rag_chain):
        self.rag_chain = rag_chain
        self.agent = IntelligentAgent(rag_chain)
        self.mode = "rag"  # 默认模式
        self.chat_history = []

    def switch_mode(self, mode_choice):
        """切换模式"""
        self.mode = mode_choice
        if mode_choice == "agent":
            return " 已切换到智能Agent模式（支持多步骤任务）"
        else:
            return " 已切换到RAG模式（纯知识库检索）"

    def respond(self, message, chat_history):
        """处理用户消息"""
        # 检查是否是切换命令
        if message.lower() in ["/agent", "/rag", "/mode agent", "/mode rag"]:
            mode = "agent" if "agent" in message.lower() else "rag"
            reply = self.switch_mode(mode)
            self.chat_history.append({"role": "user", "content": message})
            self.chat_history.append({"role": "assistant", "content": reply})
            return self.chat_history

        # 添加用户消息
        self.chat_history.append({"role": "user", "content": message})

        try:
            # 根据模式处理
            if self.mode == "agent":
                answer = self.agent.execute(message)
            else:
                answer = self.rag_chain.invoke(message)

            self.chat_history.append({"role": "assistant", "content": answer})

        except Exception as e:
            error_msg = f"系统错误: {str(e)}"
            self.chat_history.append({"role": "assistant", "content": error_msg})

        return self.chat_history

    def clear_chat(self):
        """清空聊天历史"""
        self.chat_history = []
        return []

    def create_interface(self):
        """创建Gradio界面"""
        with gr.Blocks(title="智能知识库问答系统") as interface:

            gr.Markdown("#  智能知识库问答系统")
            gr.Markdown("支持多步骤任务的智能Agent")

            with gr.Row():
                with gr.Column(scale=3):
                    # 聊天区域
                    chatbot = gr.Chatbot(
                        height=500,
                        label="对话记录",
                        value=self.chat_history
                    )

                    # 输入区域
                    with gr.Row():
                        msg = gr.Textbox(
                            placeholder="输入问题或命令（/agent 切换到Agent模式，/rag 切换到RAG模式）...",
                            show_label=False,
                            scale=4,
                            lines=2
                        )
                        submit_btn = gr.Button("发送", variant="primary", scale=1)

                    # 控制按钮
                    with gr.Row():
                        clear_btn = gr.Button("清空对话", variant="secondary")

                with gr.Column(scale=1):
                    # 模式切换区域
                    gr.Markdown("###  模式控制")

                    # 模式选择器
                    mode_radio = gr.Radio(
                        choices=[" RAG模式", " Agent模式"],
                        value=" RAG模式",
                        label="选择工作模式",
                        interactive=True
                    )

                    # 模式信息
                    mode_info = gr.Markdown("""
                     RAG模式：纯知识库检索，严格基于文档回答

                     Agent模式：
                    - 自动识别任务类型
                    - 支持多步骤任务
                    - 智能工具选择
                    """)

                    # 模式切换按钮
                    def update_mode(choice):
                        mode = "agent" if "Agent" in choice else "rag"
                        reply = self.switch_mode(mode)
                        return reply

                    mode_radio.change(
                        fn=update_mode,
                        inputs=mode_radio,
                        outputs=msg
                    )

                    # 多步骤任务示例
                    gr.Markdown("### 多步骤任务示例")

                    multistep_examples = [
                        "先计算一下5+5等于多少，然后再介绍一下神经网络",
                        "第一步：计算25*4+18等于多少，第二步：解释什么是注意力机制",
                        "先总结Transformer的核心思想，再计算3的平方根",
                        "计算(10+5)*2等于多少，然后翻译成英文",
                        "先告诉我什么是梯度下降，再计算一下15的平方",
                    ]

                    for example in multistep_examples:
                        btn = gr.Button(example[:30] + "..." if len(example) > 30 else example, size="sm")
                        btn.click(lambda q=example: q, None, msg)

                    gr.Markdown("### 单步任务示例")

                    single_examples = [
                        "神经网络的概念",
                        "计算一下100除以4等于多少",
                        "总结一下Transformer的核心思想",
                    ]

                    for example in single_examples:
                        btn = gr.Button(example[:20] + "..." if len(example) > 20 else example, size="sm")
                        btn.click(lambda q=example: q, None, msg)

            # 事件绑定
            submit_btn.click(
                fn=self.respond,
                inputs=[msg, chatbot],
                outputs=[chatbot]
            ).then(lambda: "", None, msg)

            msg.submit(
                fn=self.respond,
                inputs=[msg, chatbot],
                outputs=[chatbot]
            ).then(lambda: "", None, msg)

            clear_btn.click(
                fn=self.clear_chat,
                inputs=None,
                outputs=[chatbot]
            )

        return interface


# 十二 主函数
def main():
    print('=' * 50)
    print('智能知识库问答系统（支持多步骤任务）')
    print('=' * 50)

    # 检查知识库
    if os.path.exists(Config.KNOWLEDGE_BASE_PATH):
        md_files = [f for f in os.listdir(Config.KNOWLEDGE_BASE_PATH) if f.endswith('.md')]
        print(f"发现 {len(md_files)} 个Markdown文件")
    else:
        print(f"知识库目录不存在")

    # 初始化知识库
    if not os.path.exists(Config.PERSIST_DIRECTORY):
        print("构建知识库...")
        vectordb = build_knowledge_base()
        if vectordb is None:
            print("知识库为空，请添加文件后重新运行")
            return
    else:
        print('加载已有知识库')
        embeddings = HuggingFaceEmbeddings(model_name=Config.EMBED_MODEL_NAME)
        vectordb = Chroma(
            persist_directory=Config.PERSIST_DIRECTORY,
            embedding_function=embeddings
        )

    # 创建RAG链
    retriever = create_retrieve(vectordb)
    rag_chain = create_rag_chain(retriever)

    print('=' * 50)
    print('系统准备就绪！')
    print('Agent模式支持：')
    print('1. 多步骤任务（先...再...）')
    print('2. 自动任务分解')
    print('3. 智能工具选择')
    print('=' * 50)

    # 创建界面
    chat_interface = ChatInterface(rag_chain)
    interface = chat_interface.create_interface()

    print("请在浏览器中访问：http://localhost:7860")
    print("试试多步骤任务，如：'先计算5+5等于多少，再介绍一下神经网络'")

    interface.launch(
        server_name="0.0.0.0",
        server_port=7860,
        share=False
    )


if __name__ == "__main__":
    main()