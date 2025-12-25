#一 导入库
import os   #操作系统接口,用于文件操作
from dotenv import load_dotenv
load_dotenv() # 这会自动从 .env 文件加载环境变量
import warnings     #导入Python的warnings模块,用于处理警告
warnings.filterwarnings('ignore',category=DeprecationWarning)   #忽略DeprecationWarning（弃用警告)
os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'    #国内镜像地址加速国内访问 Hugging Face 模型和数据集

from langchain_text_splitters import RecursiveCharacterTextSplitter  #文本分割器,用于将长文档分割成小块,便于模型处理和检索,Recursive(递归)意味着它会智能地按照层次结构分割文本
from langchain_community.document_loaders import TextLoader,PyPDFLoader,DirectoryLoader   #文档加载器:支持TXT,PDF等多种格式
from langchain_huggingface import HuggingFaceEmbeddings    #嵌入模型:将文本转换为数值向量表示,用于相似性计算
from langchain_chroma import Chroma #向量数据库:存储和检索嵌入向量,存储所有文本块的向量,快速查找相似内容
from langchain_core.prompts import PromptTemplate    #提示词模板,定义如何组织问题,上下文和指令.创建标准化的提示词,提高模型回答质量
from langchain_core.runnables import RunnablePassthrough,RunnableLambda   #LangChain的流程控制组件, RunnablePassthrough:将输入原封不动传递给下一步,在RAG链中传递用户问题
from langchain_core.output_parsers import StrOutputParser  #输出解析器, StrOutputParser:将模型输出解析为字符串,确保最终输出的是纯文本格式(不然可能是AImessa(内容)的形式)
import requests #HTTP请求库,调用MiniMax API接口
import json #json数据处理库,处理API请求和响应的JSON数据
import gradio as gr   #导入gradio用于构建Web界面。
import re             #正则表达式


#二 配置信息类
class Config:
    '''
    集中管理所有配置参数,便于统一管理
    '''
    KNOWLEDGE_BASE_PATH = './knowledge_base'  #定义知识库文件路径,需要在导入库时指定类型
    PERSIST_DIRECTORY = './chroma_db'         #向量数据库的保存目录,下次启动时可直接加载,无需重新构建
    EMBED_MODEL_NAME = 'BAAI/bge-small-zh'    #嵌入模型名称,北京智源研究院的中文小模型,专门为中文优化的嵌入模型
    #MiniMax API配置
    # 从环境变量读取
    MM_API_KEY = os.environ.get('MINIMAX_API_KEY', '') # 如果没找到环境变量，则返回空字符串
    MM_GROUP_ID = os.environ.get('MINIMAX_GROUP_ID', '')
    MM_API_URL = "https://api.minimaxi.com/v1/text/chatcompletion_v2"   #API地址

#三 辅助函数(这里用来清洗数据)
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

#四 构建知识库函数
def build_knowledge_base():
    '''加载,分割文档,并创建向量数据库'''
    print('开始构建知识库...')

    #1.检查文件是否存在
    dir_path=Config.KNOWLEDGE_BASE_PATH    #获取配置中的文件路径

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

    #3.加载文档
    documents = loader.load()   #loader.load():执行文档加载,返回一个文档对象列表,PDF文档每页为一个Document对象，TXT文档整个文件为一个Document对象
    print(f"已加载文档，共 {len(documents)} 个.md文件")
    if len(documents) == 0:
        print("知识库目录中没有Markdown文件")
        print(f"请将笔记复制到 {dir_path} 目录中")
        return None

    # 4. 清洗Markdown内容（处理内部链接、图片等）
    documents = clean_markdown_content(documents)
    print("已完成Markdown内容清洗")

    #5.创建文本分割器,分割文本为小块,便于模型处理(LLM有输入长度限制)
    text_splitter = RecursiveCharacterTextSplitter(     #recursive递归,按层智能切割
        chunk_size = 500,   #每个文本块最多500字符
        chunk_overlap = 50,  #块之间的重叠字符,保持上下文,相邻块重叠50字符，防止信息割裂
        separators=["\n\n", "\n# ", "\n## ", "\n### ", "\n", "。", "，", " ", ""]  # 优先按段落和标题分割
    )
    splits = text_splitter.split_documents(documents)   #执行分割,返回更小的Document对象列表
    print(f"文档已分割为 {len(splits)} 个文本块")

    #6.创建嵌入模型(将文本转换为向量),
    # 加载预训练的中文嵌入模型
    embeddings = HuggingFaceEmbeddings(
        model_name = Config.EMBED_MODEL_NAME,
        model_kwargs = {'device': 'cpu'},   # 使用CPU，GPU可改为 'cuda'
        encode_kwargs={'normalize_embeddings': True}    #normalize_embeddings=True：归一化嵌入向量使所有向量长度为1，便于余弦相似度计算. 大概意思就是把那些影响较大的因素的影响变小
    )

    #7.创建向量数据库
    try:
        vectordb = Chroma.from_documents(
            documents = splits,     #输入分割后的文本块
            embedding = embeddings,    #使用指定的嵌入模型
            persist_directory = Config.PERSIST_DIRECTORY    #将当前内存中的向量数据库（包括索引、向量数据、元数据等）保存到指定目录
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
    return vectordb     #返回向量数据库对象供后续使用

#五 创建检索器函数
def create_retrieve(vectordb):
    """创建检索器，负责从向量库中找出与问题相关的文本块"""
    #搜索最相关的6个文本块
    retriever = vectordb.as_retriever(
        search_kwargs = {'k':6}     #k=x:f返回最相似的x个文本块,需个块平衡回答质量与处理时间(这个回答会很慢,块太少回答质量很差)
    )
    return retriever

#五,MiniMax LLM封装类
class MiniMaxLLM:
    """封装MiniMax API调用"""
    @staticmethod   #静态方法,不用创建类实例即可调用
    def invoke(prompt:str) -> str:     #类型提示：输入str，返回str
        """调用MiniMax API生成回复"""
        # 处理不同格式的提示词输入
        if hasattr(prompt, 'to_string'):    #hasattr 函数用于检查对象是否具有指定的属性或方法。它接受两个参数：对象和属性名，并返回一个布尔值。
            # 检查prompt是否有to_string方法（可能是PromptTemplate对象）
            prompt_content = prompt.to_string()
        elif not isinstance(prompt, str):
            # 如果不是字符串类型，转换为字符串
            prompt_content = str(prompt)
        else:
            prompt_content = prompt  # 已经是字符串，直接使用

        #通常在使用API时，需要设置请求头（headers），以提供必要的认证信息和指定请求体的格式
        headers = {
            'Authorization': f'Bearer {Config.MM_API_KEY}',     #Authorization: HTTP标准认证头,Bearer令牌认证: 一种认证方案（类似"钥匙"
            'Content-Type': 'application/json',  #Content-Type: 告诉服务器请求体的格式,application/json: 表示数据是JSON格式
            "Group-Id": Config.MM_GROUP_ID,
        }
        #请求体配置
        payload = {
            'model':"abab5.5-chat",    #指定模型版本
            'messages':[
                {
                    'role':'system',
                    'content':"你是一个专业的助手，严格根据提供的资料回答问题。如果资料中没有相关信息，请直接说明'根据资料无法回答此问题'，不要编造信息。"
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

#七 构建RAG应用链
def create_rag_chain(retriever):
    #定义提示词模板
    template = """请严格依据以下提供的背景资料来回答问题。如果资料中没有相关信息，请直接说明"根据资料无法回答此问题"，不要编造信息。

    **特别指令**：如果用户的问题是要求总结、概述或寻找主题，请你仔细分析所有提供的资料，进行归纳、分类和概括，梳理出清晰的结构。
    
    背景资料：
    {context}  # 占位符：将被检索到的文档替换

    问题：{question}  # 占位符：将被用户问题替换

    请基于资料提供准确、详细的回答："""

    #创建promptTemplate对象
    prompt = PromptTemplate.from_template(template)     #从template模板字符串创建提示词模板

    #定义文档格式化函数
    def format_docs(docs):
        #将Document对象列表连接为单个字符串
        return '\n\n'.join([doc.page_content for doc in docs])  ## 列表推导式：提取每个Document的page_content

    #使用LangChain表达式语言(LCEL)构建处理链,检索 -> 格式化 -> 生成
    rag_chain = (
        #1.准备输入数据字典
        {'context':retriever | format_docs,'question':RunnablePassthrough()}  #retriever|format_docs先检索再格式化
        | prompt                #填充提示词模板
        | MiniMaxLLM.invoke     #调用LLM根据提示词生成回答
        |StrOutputParser()
    )
    return rag_chain

#八 Gradio界面类(Web前端构建)
class ChatInterface:
    def __init__(self, rag_chain):
        self.rag_chain = rag_chain  # 保存RAG链
        self.chat_history = []  # 存储对话历史

    def add_message(self, role, content):
        """添加消息到聊天历史"""
        self.chat_history.append({"role": role, "content": content})
        # Gradio使用字典格式的消息：[{"role": "user", "content": "..."}, ...]

    def respond(self, message, chat_history):
        """处理用户消息并返回响应"""
        # 添加用户消息到历史
        self.add_message("user", message)

        try:
            # 显示思考状态（提升用户体验）
            thinking_msg = "正在思考..."
            self.add_message("assistant", thinking_msg)

            # 获取实际回答（调用RAG链）
            answer = self.rag_chain.invoke(message)
            # invoke：执行LangChain链

            # 更新最后一条消息为实际回答
            self.chat_history[-1] = {"role": "assistant", "content": answer}

            # 返回更新后的聊天历史
            return self.chat_history
        except Exception as e:
            # 异常处理
            error_msg = f"系统错误: {str(e)}"
            self.add_message("assistant", error_msg)
            return self.chat_history

    def clear_chat(self):
        """清空聊天历史"""
        self.chat_history = []
        return []  # 返回空列表，Gradio会用此更新聊天界面

    def create_interface(self):
        """创建Gradio界面"""
        with gr.Blocks(title="智能知识库问答系统") as interface:
            # gr.Blocks：创建块状布局界面
            # title：浏览器标签页标题

            # 标题区域
            gr.Markdown("# 智能知识库问答系统")
            # Markdown渲染，支持富文本格式
            gr.Markdown("基于个人数据库的问答助手，支持语义搜索和精准回答")

            with gr.Row():  # 创建水平行布局
                with gr.Column(scale=3):  # 左侧列，权重3（占3/4宽度）
                    # 聊天机器人组件
                    chatbot = gr.Chatbot(
                        height=500,  # 固定高度500像素
                        label="对话记录",  # 组件标签
                        value=self.chat_history  # 初始值
                    )

                    # 输入区域（嵌套Row）
                    with gr.Row():
                        msg = gr.Textbox(
                            placeholder="请输入您的问题...",  # 提示文本
                            show_label=False,  # 不显示标签
                            scale=4,  # 文本框占4份宽度
                            container=False,  # 不包含外框
                            lines=2  # 默认显示2行高度
                        )
                        submit_btn = gr.Button("发送", variant="primary", scale=1)
                        # variant="primary"：主要按钮样式

                    # 功能按钮行
                    with gr.Row():
                        clear_btn = gr.Button("清空对话", variant="secondary")
                        # variant="secondary"：次要按钮样式

                with gr.Column(scale=1):  # 右侧列，权重1（占1/4宽度）
                    # 信息面板
                    gr.Markdown("系统信息")

                    # 统计知识库文件数量
                    md_file_count = 0
                    file_list_text = ""
                    if os.path.exists(Config.KNOWLEDGE_BASE_PATH):
                        md_files = []
                        for root, dirs, files in os.walk(Config.KNOWLEDGE_BASE_PATH):
                            for file in files:
                                if file.endswith('.md'):
                                    md_files.append(file)
                                    md_file_count += 1

                        # 显示前几个文件名作为参考
                        if md_files:
                            file_list_text = "\n**当前笔记文件：**\n"
                            for i, file_name in enumerate(md_files[:5]):  # 只显示前5个
                                file_list_text += f"- {file_name}\n"
                            if len(md_files) > 5:
                                file_list_text += f"- ... 还有 {len(md_files) - 5} 个文件\n"
                    info_text = gr.Markdown(f"""
                    **模型信息**
                    - 嵌入模型：{Config.EMBED_MODEL_NAME}
                    - LLM模型：MiniMax abab5.5-chat
                    - 向量库：ChromaDB
                    **知识库**
                    - 笔记目录：{Config.KNOWLEDGE_BASE_PATH}
                    - Markdown文件数：{md_file_count} 个
                    {file_list_text}                
                    
                    **状态：**
                    - {'系统已就绪' if md_file_count > 0 else ' 等待文件...'}
                    """)

                    # 示例问题区域
                    gr.Markdown("试试这些问题：")

                    # 动态创建示例问题按钮
                    examples = [
                        "神经网络的概念",
                        "为什么要分词和编码",
                        "概括下注意力头数的作用",
                        "深度学习模型训练的流程",
                        "Git 协作开发的基本规则",
                        "Self-Attention中Q、K、V的概念",
                        "概括下前向和反向传播流程",
                        "概括下为啥梯度爆炸会导致ReLU神经元死亡",
                        "概括下进程和线程的区别"
                    ]

                    for example in examples:  # 遍历示例列表
                        # 创建按钮，文本截断处理
                        btn = gr.Button(
                            example[:25] + "..." if len(example) > 25 else example,
                            size="sm",  # 小尺寸按钮
                            variant="secondary"  # 次要样式
                        )

                        # 闭包技巧：为每个按钮创建独立的事件处理函数
                        def create_click_handler(q=example):
                            def handler():
                                return q  # 返回对应的问题文本

                            return handler

                        # 绑定点击事件
                        btn.click(
                            fn=create_click_handler(),  # 事件处理函数
                            inputs=None,  # 无输入参数
                            outputs=msg  # 输出到消息文本框
                        )

            # 事件绑定（交互逻辑）
            # 发送按钮点击事件
            submit_btn.click(
                fn=self.respond,  # 调用respond方法
                inputs=[msg, chatbot],  # 输入：消息和聊天历史
                outputs=[chatbot]  # 输出：更新后的聊天历史
            ).then(
                lambda: "", None, msg  # 清空输入框
                # then：在click之后执行
                # lambda: ""：返回空字符串的匿名函数
                # None：无输入
                # msg：输出目标（文本框）
            )

            # 消息输入框回车事件（与按钮相同功能）
            msg.submit(
                fn=self.respond,
                inputs=[msg, chatbot],
                outputs=[chatbot]
            ).then(
                lambda: "", None, msg
            )

            # 清空对话按钮点击事件
            clear_btn.click(
                fn=self.clear_chat,
                inputs=None,
                outputs=[chatbot]
            )

        return interface  # 返回创建好的界面对象

#九 主函数(程序入口)
def main():
    #标题
    print('='*50)
    print('个人知识库问答系统(RAG)')
    print('='*50)

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

    #初始化知识库
    if not os.path.exists(Config.PERSIST_DIRECTORY):
        #检查向量数据库是否存在
        print("未找到已构建的知识库，开始初始化...")
        vectordb = build_knowledge_base()   #调用函数构建向量知识库
    else:
        print('加载已有知识库')
        embeddings = HuggingFaceEmbeddings(model_name=Config.EMBED_MODEL_NAME)
        vectordb = Chroma(
            persist_directory=Config.PERSIST_DIRECTORY,     #向量数据存储位置
            embedding_function=embeddings                   #使用指定模型进行向量化
        )

    #创建RAG链
    retriever = create_retrieve(vectordb)      #从向量数据库创建检索器
    rag_chain = create_rag_chain(retriever)     #创建完整的RAG处理链

    print('知识库准备就绪')
    print('正在启动Web界面...')

    # 创建并启动Gradio界面
    chat_interface = ChatInterface(rag_chain)  # 实例化界面类
    interface = chat_interface.create_interface()  # 创建界面

    print("请在浏览器中访问：http://localhost:7860")

    # 启动Gradio服务器
    interface.launch(
        server_name="0.0.0.0",  # 监听所有网络接口
        server_port=7860,  # 端口号
        share=False,  # 不创建公共链接
        show_error=True,  # 显示错误信息
        quiet=False,  # 显示启动信息
        theme=gr.themes.Soft()  # 使用Soft主题
    )


# 程序入口点
if __name__ == "__main__":
    main()
    # __name__ == "__main__"：判断是否直接运行此脚本
    # 如果是导入此脚本，则不执行main()
