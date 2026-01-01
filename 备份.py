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
import hashlib
import time
from datetime import datetime
from typing import List, Dict, Any, Optional


# 二 配置信息类
class Config:
    KNOWLEDGE_BASE_PATH = './knowledge_base'
    PERSIST_DIRECTORY = './chroma_db'
    EMBED_MODEL_NAME = 'BAAI/bge-small-zh'
    MM_API_KEY = os.environ.get('MINIMAX_API_KEY', '')
    MM_GROUP_ID = os.environ.get('MINIMAX_GROUP_ID', '')
    MM_API_URL = "https://api.minimaxi.com/v1/text/chatcompletion_v2"
    LLM_CLASSIFIER_CACHE_SIZE = 500  # 分类器缓存大小


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


# 六、MiniMax LLM封装类
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


# 八、LLM任务分类器（新增）
class LLMTaskClassifier:
    """使用LLM进行智能任务分类"""

    def __init__(self, llm_api, cache_size=500, use_cache=True):
        self.llm_api = llm_api
        self.use_cache = use_cache
        self.cache = {}
        self.cache_size = cache_size
        self.cache_hits = 0
        self.cache_misses = 0

        # 预定义分类（避免频繁调用LLM）
        self.predefined_classifications = {
            # 计算任务
            "计算一下5+5等于多少": "calculation",
            "5+5等于几": "calculation",
            "计算25乘以4": "calculation",
            "12的平方是多少": "calculation",

            # 总结任务
            "总结这篇文章": "summary",
            "概括主要内容": "summary",
            "列出要点": "summary",

            # 不需要检索的闲聊
            "你好": "direct_chat",
            "你好啊": "direct_chat",
            "早上好": "direct_chat",
            "晚上好": "direct_chat",
            "谢谢": "direct_chat",
            "再见": "direct_chat",
            "今天天气怎么样": "direct_chat",
            "讲个笑话": "direct_chat",

            # 明确需要检索的知识查询
            "什么是神经网络": "rag_query",
            "介绍一下Transformer": "rag_query",
            "解释梯度下降": "rag_query",
            "Python列表的用法": "rag_query",
        }

        # 知识库主题
        self.kb_topics = [
            "神经网络", "深度学习", "机器学习", "人工智能",
            "Python", "编程", "算法", "数据结构",
            "Transformer", "注意力机制", "BERT", "GPT",
            "梯度下降", "反向传播", "损失函数", "优化器",
        ]

    def classify(self, query: str, use_fallback=True) -> Dict[str, Any]:
        """使用LLM进行任务分类"""
        # 1. 预处理查询
        cleaned_query = self._preprocess_query(query)

        # 2. 检查预定义分类
        predefined_result = self._check_predefined_classification(cleaned_query)
        if predefined_result:
            return {
                "classification": predefined_result,
                "confidence": 0.95,
                "source": "predefined",
                "explanation": "匹配预定义分类规则"
            }

        # 3. 检查缓存
        cache_key = self._generate_cache_key(cleaned_query)
        if self.use_cache and cache_key in self.cache:
            self.cache_hits += 1
            return self.cache[cache_key]

        # 4. 使用LLM分类
        self.cache_misses += 1
        llm_result = self._classify_with_llm(cleaned_query)

        # 5. 处理LLM失败的情况
        if llm_result.get("classification") == "error" and use_fallback:
            fallback_result = self._fallback_classification(cleaned_query)
            llm_result = fallback_result

        # 6. 更新缓存
        if self.use_cache:
            self._update_cache(cache_key, llm_result)

        return llm_result

    def _preprocess_query(self, query: str) -> str:
        """预处理查询文本"""
        cleaned = ' '.join(query.split())
        cleaned = re.sub(r'[？?]+', '?', cleaned)
        cleaned = re.sub(r'[！!]+', '!', cleaned)

        if len(cleaned) > 500:
            cleaned = cleaned[:497] + "..."

        return cleaned

    def _check_predefined_classification(self, query: str) -> Optional[str]:
        """检查预定义分类"""
        if query in self.predefined_classifications:
            return self.predefined_classifications[query]

        query_lower = query.lower()

        # 基于关键词的部分匹配
        patterns = [
            (["计算", "等于", "多少", "加", "减", "乘", "除", "平方", "立方"], "calculation"),
            (["总结", "概括", "概述", "要点", "主要观点"], "summary"),
            (["翻译", "译成", "英文", "中文", "日语"], "translation"),
            (["你好", "hi", "hello", "早上好", "晚上好", "再见", "拜拜"], "direct_chat"),
        ]

        for keywords, classification in patterns:
            if any(keyword in query_lower for keyword in keywords):
                if classification == "calculation":
                    if any(char.isdigit() for char in query):
                        return classification
                else:
                    return classification

        return None

    def _generate_cache_key(self, query: str) -> str:
        """生成缓存键"""
        query_normalized = query.lower().strip()
        return hashlib.md5(query_normalized.encode()).hexdigest()[:12]

    def _classify_with_llm(self, query: str) -> Dict[str, Any]:
        """使用LLM进行分类"""
        prompt = self._build_classification_prompt(query)

        try:
            start_time = time.time()
            response = self.llm_api(prompt)
            elapsed_time = time.time() - start_time

            if elapsed_time > 5.0:
                print(f"LLM分类超时: {elapsed_time:.2f}秒")

            return self._parse_llm_response(response, query)

        except Exception as e:
            print(f"LLM分类失败: {str(e)}")
            return {
                "classification": "error",
                "confidence": 0.0,
                "source": "llm_error",
                "explanation": f"LLM分类失败: {str(e)}"
            }

    def _build_classification_prompt(self, query: str) -> str:
        """构建分类提示词"""
        kb_topics_str = "知识库主要包含以下主题：\n" + ", ".join(self.kb_topics[:10]) + "\n\n"

        prompt = f"""你是一个任务分类助手。请分析用户的问题，判断它属于以下哪种任务类型：

## 任务类型定义：
1. **calculation** - 数学计算问题，包含数字和数学运算
   示例："计算5+5"、"12的平方是多少"、"25乘以4加18等于多少"

2. **summary** - 文本总结问题，要求概括或提炼要点
   示例："总结这篇文章"、"概括一下主要内容"、"列出三个要点"

3. **translation** - 翻译任务，要求将文本从一种语言翻译到另一种语言
   示例："翻译成英文"、"把这段话译成中文"、"英文翻译"

4. **rag_query** - 需要基于特定知识库回答的问题
   示例："什么是神经网络"、"介绍一下Transformer"、"Python列表怎么排序"

5. **direct_chat** - 通用闲聊、问候或常识问题，不需要特定知识库
   示例："你好"、"今天天气怎么样"、"讲个笑话"、"谢谢"

## 判断规则：
- 如果问题中包含数字和数学运算符，优先考虑calculation
- 如果问题要求总结、概括或提炼要点，考虑summary
- 如果问题明确要求翻译，考虑translation
- 如果问题涉及技术概念、专业知识或需要特定文档，考虑rag_query
- 如果问题是简单闲聊、问候或通用常识，考虑direct_chat

{kb_topics_str}
## 用户问题：
"{query}"

## 请严格按以下JSON格式回答，不要添加其他内容：
{{
  "classification": "任务类型名称",
  "confidence": 0.0到1.0的置信度,
  "explanation": "简要解释为什么这样分类"
}}

请确保classification字段的值只能是：calculation, summary, translation, rag_query, direct_chat 中的一个。"""

        return prompt

    def _parse_llm_response(self, response: str, original_query: str) -> Dict[str, Any]:
        """解析LLM响应"""
        cleaned_response = response.strip()

        # 尝试提取JSON部分
        json_match = re.search(r'\{.*\}', cleaned_response, re.DOTALL)

        if json_match:
            try:
                json_str = json_match.group(0)
                result = json.loads(json_str)

                # 验证分类类型
                valid_classifications = ["calculation", "summary", "translation", "rag_query", "direct_chat"]
                classification = result.get("classification", "").lower()

                if classification not in valid_classifications:
                    return self._fallback_classification(original_query)

                confidence = float(result.get("confidence", 0.5))
                confidence = max(0.0, min(1.0, confidence))

                return {
                    "classification": classification,
                    "confidence": confidence,
                    "source": "llm",
                    "explanation": result.get("explanation", "")
                }

            except json.JSONDecodeError:
                pass

        # 直接查找分类关键词
        direct_match = re.search(
            r'(calculation|summary|translation|rag_query|direct_chat)',
            cleaned_response,
            re.IGNORECASE
        )

        if direct_match:
            classification = direct_match.group(0).lower()
            confidence = 0.7 if "confidence" in cleaned_response.lower() else 0.6

            return {
                "classification": classification,
                "confidence": confidence,
                "source": "llm_fallback",
                "explanation": "从响应文本中直接提取的分类"
            }

        # 使用回退策略
        return self._fallback_classification(original_query)

    def _fallback_classification(self, query: str) -> Dict[str, Any]:
        """回退分类策略"""
        query_lower = query.lower()

        # 基于规则的简单分类
        if any(char.isdigit() for char in query):
            if any(op in query_lower for op in ["加", "减", "乘", "除", "+", "-", "*", "/"]):
                return {
                    "classification": "calculation",
                    "confidence": 0.7,
                    "source": "fallback_rule",
                    "explanation": "包含数字和运算符，推测为计算任务"
                }

        if any(word in query_lower for word in ["总结", "概括", "概述", "要点"]):
            return {
                "classification": "summary",
                "confidence": 0.7,
                "source": "fallback_rule",
                "explanation": "包含总结相关词汇"
            }

        if any(word in query_lower for word in ["翻译", "译成", "英文", "中文"]):
            return {
                "classification": "translation",
                "confidence": 0.7,
                "source": "fallback_rule",
                "explanation": "包含翻译相关词汇"
            }

        if any(word in query_lower for word in ["你好", "hi", "hello", "天气", "笑话"]):
            return {
                "classification": "direct_chat",
                "confidence": 0.6,
                "source": "fallback_rule",
                "explanation": "看起来像是闲聊或常识问题"
            }

        # 默认：去知识库检索
        return {
            "classification": "rag_query",
            "confidence": 0.5,
            "source": "fallback_default",
            "explanation": "无法确定类型，默认使用知识库检索"
        }

    def _update_cache(self, cache_key: str, result: Dict[str, Any]):
        """更新缓存"""
        if len(self.cache) >= self.cache_size:
            if self.cache:
                oldest_key = next(iter(self.cache))
                del self.cache[oldest_key]

        self.cache[cache_key] = result

    def get_cache_stats(self) -> Dict[str, Any]:
        """获取缓存统计信息"""
        total = self.cache_hits + self.cache_misses
        hit_rate = self.cache_hits / total if total > 0 else 0

        return {
            "cache_size": len(self.cache),
            "cache_hits": self.cache_hits,
            "cache_misses": self.cache_misses,
            "hit_rate": hit_rate
        }

    def clear_cache(self):
        """清空缓存"""
        self.cache.clear()
        self.cache_hits = 0
        self.cache_misses = 0


# 九、增强版任务规划器（替换原有TaskPlanner）
class EnhancedTaskPlanner:
    """增强版任务规划器（支持LLM智能分类）"""

    def __init__(self, llm_classifier=None, mode="hybrid"):
        """
        初始化增强版任务规划器

        Args:
            llm_classifier: LLM任务分类器实例
            mode: 工作模式，可选 "rule", "llm", "hybrid"
        """
        self.mode = mode
        self.llm_classifier = llm_classifier

        # 原有的规则分类器关键词
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

        # 统计信息
        self.stats = {
            "total_queries": 0,
            "rule_decisions": 0,
            "llm_decisions": 0,
            "multistep_tasks": 0,
            "single_step_tasks": 0
        }

    def set_mode(self, mode: str) -> str:
        """设置工作模式"""
        valid_modes = ["rule", "llm", "hybrid"]
        if mode not in valid_modes:
            raise ValueError(f"模式必须是 {valid_modes} 之一")
        self.mode = mode
        return f"任务规划器模式已设置为: {mode}"

    def is_multistep_task(self, query: str) -> bool:
        """判断是否是多步骤任务"""
        self.stats["total_queries"] += 1

        query_lower = query.lower()
        keyword_count = 0
        for keyword in self.multistep_keywords:
            if keyword in query_lower:
                keyword_count += 1
                if keyword_count >= 2:
                    self.stats["multistep_tasks"] += 1
                    return True

        if "第一步" in query_lower and ("第二步" in query_lower or "然后" in query_lower):
            self.stats["multistep_tasks"] += 1
            return True

        if "先" in query_lower and ("再" in query_lower or "然后" in query_lower):
            self.stats["multistep_tasks"] += 1
            return True

        self.stats["single_step_tasks"] += 1
        return False

    def identify_steps(self, query: str) -> List[str]:
        """识别任务步骤"""
        steps = []

        # 如果启用了LLM模式，尝试用LLM分解复杂任务
        if self.mode in ["llm", "hybrid"] and self.llm_classifier:
            if self._is_complex_task(query):
                llm_steps = self._decompose_with_llm(query)
                if llm_steps and len(llm_steps) > 1:
                    return llm_steps

        # 使用原有的规则分解
        query_lower = query.lower()

        # 尝试按常见模式分割
        if "先" in query_lower and "再" in query_lower:
            parts = query.split("先")[1].split("再")
            if len(parts) >= 2:
                steps.append(f"先{parts[0]}".strip())
                steps.append(f"再{parts[1]}".strip())
        elif "然后" in query_lower:
            parts = query.split("然后")
            steps = [p.strip() for p in parts if p.strip()]
        elif "接着" in query_lower:
            parts = query.split("接着")
            steps = [p.strip() for p in parts if p.strip()]
        elif "第一步" in query_lower:
            step_pattern = r'第[一二三四五]步[:：]?\s*(.*?)(?=第[一二三四五]步|$)'
            matches = re.findall(step_pattern, query)
            if matches:
                steps = [m.strip() for m in matches if m.strip()]

        # 如果以上模式都没匹配到，尝试按标点分割
        if not steps:
            sentences = re.split(r'[。；;，,]', query)
            steps = [s.strip() for s in sentences if s.strip() and len(s.strip()) > 3]

        # 清理步骤文本
        cleaned_steps = []
        for step in steps:
            step_clean = step
            for keyword in self.multistep_keywords:
                if step_clean.startswith(keyword):
                    step_clean = step_clean[len(keyword):].strip()
            cleaned_steps.append(step_clean)

        return cleaned_steps if cleaned_steps else [query]

    def classify_step(self, step: str) -> Dict[str, Any]:
        """分类单个步骤的类型（增强版）"""
        classification_result = None

        # 根据模式选择分类器
        if self.mode == "rule":
            # 纯规则模式
            classification = self._rule_classify_step(step)
            classification_result = {
                "classification": classification,
                "confidence": 0.7 if classification != "rag_query" else 0.5,
                "source": "rule_based",
                "explanation": "基于规则分类"
            }
            self.stats["rule_decisions"] += 1

        elif self.mode == "llm" and self.llm_classifier:
            # 纯LLM模式
            classification_result = self.llm_classifier.classify(step)
            self.stats["llm_decisions"] += 1

        else:  # hybrid模式
            # 混合模式：先用规则，如果不确定再用LLM
            rule_classification = self._rule_classify_step(step)

            # 计算规则方法的置信度
            rule_confidence = self._calculate_rule_confidence(step, rule_classification)

            if rule_confidence > 0.8:
                # 规则置信度高，直接使用
                classification_result = {
                    "classification": rule_classification,
                    "confidence": rule_confidence,
                    "source": "rule_based",
                    "explanation": "基于规则分类（高置信度）"
                }
                self.stats["rule_decisions"] += 1
            elif self.llm_classifier:
                # 规则不确定，使用LLM
                classification_result = self.llm_classifier.classify(step)
                self.stats["llm_decisions"] += 1
            else:
                # 没有LLM分类器，使用规则结果
                classification_result = {
                    "classification": rule_classification,
                    "confidence": rule_confidence,
                    "source": "rule_based",
                    "explanation": "基于规则分类（低置信度）"
                }
                self.stats["rule_decisions"] += 1

        return classification_result or {
            "classification": "rag_query",
            "confidence": 0.5,
            "source": "fallback",
            "explanation": "分类失败，使用默认检索"
        }

    def _rule_classify_step(self, step: str) -> str:
        """原有规则分类方法"""
        step_lower = step.lower()

        if any(keyword in step_lower for keyword in self.calculation_keywords):
            if any(char.isdigit() for char in step):
                return "calculation"

        if any(keyword in step_lower for keyword in self.summary_keywords):
            return "summary"

        return "rag_query"

    def _is_complex_task(self, query: str) -> bool:
        """判断是否为复杂任务"""
        query_lower = query.lower()

        action_verbs = ["计算", "总结", "翻译", "介绍", "解释", "比较", "分析"]
        verb_count = sum(1 for verb in action_verbs if verb in query_lower)

        is_long = len(query) > 50
        comma_count = query.count(',') + query.count('，')

        return verb_count >= 2 or is_long or comma_count >= 2

    def _decompose_with_llm(self, query: str) -> List[str]:
        """使用LLM分解复杂任务"""
        if not self.llm_classifier:
            return []

        # 检查是否需要分解
        query_lower = query.lower()

        # 对于"介绍一下"、"解释一下"这类简单查询，不进行分解
        simple_patterns = [
            "介绍一下", "解释一下", "什么是", "谈谈",
            "讲一下", "说明一下", "描述一下"
        ]

        if any(pattern in query_lower for pattern in simple_patterns):
            # 简单查询，不分解
            return []

        prompt = f"""请判断以下任务是否需要分解为多个子步骤：

    任务：{query}

    判断标准：
    1. 如果任务本身已经很明确且具体，不需要分解
    2. 如果任务包含多个明显不同的部分（如"先计算...再介绍..."），需要分解
    3. 如果任务是单一的、完整的查询，不需要分解

    请用JSON格式回答：
    {{
        "need_decomposition": true/false,
        "steps": ["步骤1", "步骤2", ...]  # 如果need_decomposition为true
    }}

    示例1：
    输入："先计算5+5，再介绍神经网络"
    输出：{{"need_decomposition": true, "steps": ["计算5+5", "介绍神经网络"]}}

    示例2：
    输入："介绍一下神经网络"
    输出：{{"need_decomposition": false, "steps": []}}

    现在请分析上述任务："""

        try:
            response = MiniMaxLLM.invoke(prompt)

            # 解析JSON响应
            try:
                result = json.loads(response)
                if result.get("need_decomposition", False):
                    steps = result.get("steps", [])
                    # 限制最大步骤数，避免过度分解
                    return steps[:3]  # 最多3个步骤
                else:
                    return []
            except json.JSONDecodeError:
                # 解析失败，保守策略：不分解
                return []

        except Exception as e:
            print(f"LLM任务分解失败: {e}")
            return []

    def _calculate_rule_confidence(self, step: str, classification: str) -> float:
        """计算规则分类的置信度"""
        confidence = 0.5
        step_lower = step.lower()

        if classification == "calculation":
            has_digit = any(char.isdigit() for char in step)
            has_operator = any(op in step_lower for op in ["加", "减", "乘", "除", "+", "-", "*", "/"])

            if has_digit and has_operator:
                confidence = 0.9
            elif has_digit:
                confidence = 0.7
            else:
                confidence = 0.4

        elif classification == "summary":
            summary_keywords = ["总结", "概括", "概述", "要点", "主要", "核心"]
            keyword_count = sum(1 for kw in summary_keywords if kw in step_lower)

            if keyword_count >= 2:
                confidence = 0.9
            elif keyword_count == 1:
                confidence = 0.7
            else:
                confidence = 0.4

        elif classification == "rag_query":
            professional_terms = ["神经网络", "机器学习", "Python", "算法", "模型", "训练"]
            term_count = sum(1 for term in professional_terms if term in step)

            if term_count >= 1:
                confidence = 0.8
            else:
                confidence = 0.5

        return confidence

    def get_stats(self) -> Dict[str, Any]:
        """获取统计信息"""
        return self.stats


# 十、增强版智能Agent（替换原有IntelligentAgent）
class EnhancedIntelligentAgent:
    """增强版智能Agent，支持LLM智能分类"""

    def __init__(self, rag_chain, task_planner=None, llm_classifier=None):
        """
        初始化智能Agent

        Args:
            rag_chain: RAG链
            task_planner: 任务规划器（如果不提供则创建默认的）
            llm_classifier: LLM分类器（可选）
        """
        self.rag_chain = rag_chain
        self.tools = ToolManager()

        # 创建或使用传入的任务规划器
        if task_planner:
            self.task_planner = task_planner
        else:
            # 创建增强版任务规划器
            if llm_classifier:
                self.task_planner = EnhancedTaskPlanner(
                    llm_classifier=llm_classifier,
                    mode="hybrid"
                )
            else:
                self.task_planner = EnhancedTaskPlanner(mode="rule")

        # 对话历史管理
        self.conversation_history = []
        self.max_history_length = 10

    def process_multistep_task(self, query: str) -> str:
        """处理多步骤任务（增强版）"""
        # 记录对话历史
        self._add_to_history("user", query)

        try:
            is_multistep = self.task_planner.is_multistep_task(query)

            if not is_multistep:
                # 单步任务
                result = self._process_single_task(query, is_multistep=False)
                self._add_to_history("assistant", result)
                return result

            # 多步骤任务
            steps = self.task_planner.identify_steps(query)

            if len(steps) <= 1:
                # 识别失败，回退到单步处理
                result = self._process_single_task(query, is_multistep=False)
                self._add_to_history("assistant", result)
                return result

            # 执行每个步骤
            results = []
            for i, step in enumerate(steps, 1):
                step_result = self._process_single_step(step, i)
                results.append(step_result)

            # 组合所有结果
            final_result = self._combine_results(results, query)
            self._add_to_history("assistant", final_result)
            return final_result

        except Exception as e:
            error_msg = f"处理任务时出错: {str(e)}"
            self._add_to_history("assistant", error_msg)
            return error_msg

    def _process_single_step(self, step: str, step_num: int) -> Dict[str, Any]:
        """处理单个步骤（使用LLM分类）"""
        # 获取步骤分类
        classification_info = self.task_planner.classify_step(step)
        step_type = classification_info["classification"]
        confidence = classification_info.get("confidence", 0.5)

        # 根据分类执行
        if step_type == "calculation" and confidence > 0.6:
            result = self.tools.calculate_tool(step)
            return {
                "type": "calculation",
                "step": step,
                "result": result,
                "step_num": step_num,
                "classification_info": classification_info
            }

        elif step_type == "summary" and confidence > 0.6:
            result = self.tools.summarize_tool(step)
            return {
                "type": "summary",
                "step": step,
                "result": result,
                "step_num": step_num,
                "classification_info": classification_info
            }

        elif step_type == "translation" and confidence > 0.6:
            target_lang = "英文"
            if "英文" in step or "英语" in step:
                target_lang = "英文"
            elif "中文" in step or "汉语" in step:
                target_lang = "中文"

            result = self.tools.translate_tool(step, target_lang)
            return {
                "type": "translation",
                "step": step,
                "result": result,
                "step_num": step_num,
                "classification_info": classification_info
            }

        elif step_type == "direct_chat" and confidence > 0.6:
            result = self._direct_chat_response(step)
            return {
                "type": "direct_chat",
                "step": step,
                "result": result,
                "step_num": step_num,
                "classification_info": classification_info
            }

        else:
            result = self.rag_chain.invoke(step)
            return {
                "type": "rag_query",
                "step": step,
                "result": result,
                "step_num": step_num,
                "classification_info": classification_info
            }

    def _process_single_task(self, query: str, is_multistep: bool = False) -> str:
        """处理单步任务"""
        classification_info = self.task_planner.classify_step(query)
        step_type = classification_info["classification"]
        confidence = classification_info.get("confidence", 0.5)

        if step_type == "calculation" and confidence > 0.6:
            result = self.tools.calculate_tool(query)
            response = f" **计算结果**\n\n{result}"

        elif step_type == "summary" and confidence > 0.6:
            result = self.tools.summarize_tool(query)
            response = f" **总结结果**\n\n{result}"

        elif step_type == "translation" and confidence > 0.6:
            target_lang = "英文"
            if "英文" in query or "英语" in query:
                target_lang = "英文"
            elif "中文" in query or "汉语" in query:
                target_lang = "中文"

            result = self.tools.translate_tool(query, target_lang)
            response = f" **翻译结果**\n\n{result}"

        elif step_type == "direct_chat" and confidence > 0.6:
            result = self._direct_chat_response(query)
            response = f" {result}"

        else:
            result = self.rag_chain.invoke(query)
            response = f" **知识库查询结果**\n\n{result}"

        if is_multistep:
            response += f"\n\n*分类: {step_type} (置信度: {confidence:.2f})*"

        return response

    def _direct_chat_response(self, query: str) -> str:
        """处理直接聊天"""
        query_lower = query.lower()

        if any(word in query_lower for word in ["你好", "hi", "hello", "您好"]):
            return "你好！我是智能助手，有什么可以帮你的吗？"

        elif any(word in query_lower for word in ["谢谢", "感谢", "多谢"]):
            return "不客气！很高兴能帮到你。"

        elif any(word in query_lower for word in ["再见", "拜拜", "88"]):
            return "再见！祝你有个愉快的一天！"

        elif "天气" in query_lower:
            return "我目前无法获取实时天气信息。你可以查看天气预报应用或网站获取最新天气。"

        elif "笑话" in query_lower or "讲个笑话" in query_lower:
            jokes = [
                "为什么程序员总是分不清万圣节和圣诞节？因为 Oct 31 == Dec 25！",
                "为什么Java程序员要戴眼镜？因为他们不会C#！",
                "我有个关于递归的笑话，但是...等等，我已经说过了。"
            ]
            return random.choice(jokes)

        else:
            prompt = f"用户说：{query}\n请给出一个友好、有帮助的回应。"
            return MiniMaxLLM.invoke(prompt)

    def _combine_results(self, results: List[Dict[str, Any]], original_query: str) -> str:
        """组合多步骤结果（增强版）"""
        final_output = " **多步骤任务执行完成**\n\n"
        final_output += f"**原始指令**: {original_query}\n\n"
        final_output += "**执行过程**:\n\n"

        for result in results:
            step_num = result.get("step_num", "?")
            step = result.get("step", "")
            step_result = result.get("result", "")
            step_type = result.get("type", "unknown")
            classification_info = result.get("classification_info", {})

            confidence = classification_info.get("confidence", 0.5)
            source = classification_info.get("source", "unknown")

            if step_type == "calculation":
                final_output += f"{step_num}.  **计算** ({source}, 置信度: {confidence:.2f}): {step}\n"
                final_output += f"   结果: {step_result}\n\n"
            elif step_type == "rag_query":
                final_output += f"{step_num}.  **知识查询** ({source}, 置信度: {confidence:.2f}): {step}\n"
                final_output += f"   结果: {step_result}\n\n"
            elif step_type == "summary":
                final_output += f"{step_num}.  **总结** ({source}, 置信度: {confidence:.2f}): {step}\n"
                final_output += f"   结果: {step_result}\n\n"
            elif step_type == "translation":
                final_output += f"{step_num}.  **翻译** ({source}, 置信度: {confidence:.2f}): {step}\n"
                final_output += f"   结果: {step_result}\n\n"
            elif step_type == "direct_chat":
                final_output += f"{step_num}.  **聊天** ({source}, 置信度: {confidence:.2f}): {step}\n"
                final_output += f"   结果: {step_result}\n\n"
            else:
                final_output += f"{step_num}.  **未知类型**: {step}\n"
                final_output += f"   结果: {step_result}\n\n"

        # 添加统计信息
        stats = self.task_planner.get_stats()
        cache_stats = {}
        if hasattr(self.task_planner, 'llm_classifier') and self.task_planner.llm_classifier:
            cache_stats = self.task_planner.llm_classifier.get_cache_stats()

        final_output += "---\n"
        final_output += f"*任务统计: 总查询{stats['total_queries']}次，"
        final_output += f"多步骤任务{stats['multistep_tasks']}次，"
        final_output += f"规则决策{stats['rule_decisions']}次，"
        final_output += f"LLM决策{stats['llm_decisions']}次*\n"

        if cache_stats:
            hit_rate = cache_stats.get('hit_rate', 0)
            hits = cache_stats.get('cache_hits', 0)
            misses = cache_stats.get('cache_misses', 0)
            final_output += f"*缓存命中率: {hit_rate:.1%} ({hits}/{hits + misses})*"

        return final_output

    def _add_to_history(self, role: str, content: str):
        """添加到对话历史"""
        self.conversation_history.append({
            "role": role,
            "content": content,
            "timestamp": datetime.now().strftime("%H:%M:%S")
        })

        if len(self.conversation_history) > self.max_history_length:
            self.conversation_history = self.conversation_history[-self.max_history_length:]

    def execute(self, query: str) -> str:
        """执行Agent决策（兼容原有接口）"""
        return self.process_multistep_task(query)

    def set_planner_mode(self, mode: str) -> str:
        """设置任务规划器模式"""
        return self.task_planner.set_mode(mode)

    def get_planner_stats(self) -> Dict[str, Any]:
        """获取任务规划器统计信息"""
        return self.task_planner.get_stats()

    def get_cache_stats(self) -> Dict[str, Any]:
        """获取缓存统计信息"""
        if hasattr(self.task_planner, 'llm_classifier') and self.task_planner.llm_classifier:
            return self.task_planner.llm_classifier.get_cache_stats()
        return {}

    def clear_cache(self) -> str:
        """清空缓存"""
        if hasattr(self.task_planner, 'llm_classifier') and self.task_planner.llm_classifier:
            self.task_planner.llm_classifier.clear_cache()
            return "缓存已清空"
        return "没有找到缓存"

    def clear_history(self):
        """清空对话历史"""
        self.conversation_history = []


# 十一、RAG链函数（保持不变）
def create_rag_chain(retriever):
    template = """请严格依据以下提供的背景资料来回答问题。

## 重要规则：
1. **严格基于资料**：只使用资料中的信息，不添加外部知识
2. **资料不足时**：如果资料中没有相关信息，请根据你的通用知识简要回答
3. **资料冲突时**：如果不同资料有冲突，请指出并说明
4. **明确引用**：如果可能，在回答中注明信息来自哪份资料
5. **结构化回答**：使用清晰的段落和列表

## 背景资料：
{context}

## 用户问题：
{question}

请基于上述要求提供准确、详细的回答。如果资料中没有相关信息，请根据你的通用知识简要回答，但请注明"注：以下信息基于通用知识，可能不完整"。"""

    prompt = PromptTemplate.from_template(template)

    def format_docs(docs):
        # 检查文档质量
        if not docs:
            return "未找到相关资料。"

        # 检查文档相关性
        formatted = []
        for i, doc in enumerate(docs):
            # 可以在这里添加文档相关性评分
            formatted.append(f"[资料{i + 1}]\n{doc.page_content}")

        return '\n\n'.join(formatted)

    rag_chain = (
            {'context': retriever | format_docs, 'question': RunnablePassthrough()}
            | prompt
            | MiniMaxLLM.invoke
            | StrOutputParser()
    )
    return rag_chain


# 十二、增强版Gradio界面类（修复兼容性问题）
class EnhancedChatInterface:
    """增强版聊天界面"""

    def __init__(self, agent, llm_classifier):
        self.agent = agent
        self.llm_classifier = llm_classifier
        self.mode = "hybrid"
        self.chat_history = []

    def switch_mode(self, mode_choice: str) -> str:
        """切换工作模式"""
        valid_modes = {"rule": "规则模式", "llm": "LLM模式", "hybrid": "混合模式"}

        if mode_choice in valid_modes:
            self.mode = mode_choice
            result = self.agent.set_planner_mode(mode_choice)
            return f" 已切换到{valid_modes[mode_choice]}\n{result}"
        else:
            return f" 无效模式。可选：{', '.join(valid_modes.keys())}"

    def show_stats(self) -> str:
        """显示系统统计"""
        planner_stats = self.agent.get_planner_stats()
        cache_stats = self.agent.get_cache_stats()

        stats_text = " **系统统计信息**\n\n"
        stats_text += f"**任务规划器统计**:\n"
        stats_text += f"- 总查询次数: {planner_stats.get('total_queries', 0)}\n"
        stats_text += f"- 多步骤任务: {planner_stats.get('multistep_tasks', 0)}\n"
        stats_text += f"- 规则决策: {planner_stats.get('rule_decisions', 0)}\n"
        stats_text += f"- LLM决策: {planner_stats.get('llm_decisions', 0)}\n\n"

        if cache_stats:
            stats_text += f"**缓存统计**:\n"
            stats_text += f"- 缓存大小: {cache_stats.get('cache_size', 0)}\n"
            stats_text += f"- 缓存命中: {cache_stats.get('cache_hits', 0)}\n"
            stats_text += f"- 缓存未命中: {cache_stats.get('cache_misses', 0)}\n"
            hit_rate = cache_stats.get('hit_rate', 0)
            stats_text += f"- 命中率: {hit_rate:.1%}\n"

        stats_text += f"\n**当前模式**: {self.mode}"

        return stats_text

    def respond(self, message: str, chat_history: List[Dict]) -> List[Dict]:
        """处理用户消息"""
        # 检查是否是命令
        if message.lower().startswith('/'):
            reply = self.handle_command(message.lower())
            self.chat_history.append({"role": "user", "content": message})
            self.chat_history.append({"role": "assistant", "content": reply})
            return self.chat_history

        # 添加用户消息
        self.chat_history.append({"role": "user", "content": message})

        try:
            # 处理查询
            answer = self.agent.execute(message)
            self.chat_history.append({"role": "assistant", "content": answer})

        except Exception as e:
            error_msg = f"系统错误: {str(e)}"
            self.chat_history.append({"role": "assistant", "content": error_msg})

        return self.chat_history

    def handle_command(self, command: str) -> str:
        """处理命令"""
        command = command.strip()

        if command.startswith('/mode'):
            mode = command.split()[-1] if len(command.split()) > 1 else "hybrid"
            return self.switch_mode(mode)

        elif command == '/stats':
            return self.show_stats()

        elif command == '/clear_cache':
            result = self.agent.clear_cache()
            return f" {result}"

        elif command == '/help':
            help_text = " **可用命令**:\n\n"
            help_text += "/mode [rule|llm|hybrid] - 切换工作模式\n"
            help_text += "/stats - 显示系统统计\n"
            help_text += "/clear_cache - 清空分类缓存\n"
            help_text += "/help - 显示此帮助信息\n\n"
            help_text += "**多步骤任务示例**:\n"
            help_text += "先计算5+5，再介绍神经网络\n"
            help_text += "总结Transformer，然后翻译成英文"
            return help_text

        elif command == '/test':
            test_queries = [
                "计算5+5等于多少",
                "总结这篇文章",
                "翻译成英文",
                "什么是神经网络",
                "你好",
                "今天天气怎么样"
            ]

            results = []
            for query in test_queries:
                if self.llm_classifier:
                    result = self.llm_classifier.classify(query)
                    results.append(f"{query}: {result['classification']} (置信度: {result['confidence']:.2f})")

            return " **分类器测试结果**:\n\n" + "\n".join(results)

        else:
            return f" 未知命令: {command}\n输入 /help 查看可用命令"

    def clear_chat(self) -> List:
        """清空聊天历史"""
        self.chat_history = []
        self.agent.clear_history()
        return []

    def create_interface(self):
        """创建Gradio界面"""
        with gr.Blocks(title="智能知识库问答系统") as interface:

            # 标题和说明
            gr.Markdown("#  智能知识库问答系统")
            gr.Markdown("支持LLM智能分类和多步骤任务处理")

            with gr.Row():
                with gr.Column(scale=3):
                    # 聊天区域 - 移除不兼容的参数
                    chatbot = gr.Chatbot(
                        height=500,
                        label="对话记录",
                        value=self.chat_history
                    )

                    # 输入区域
                    with gr.Row():
                        msg = gr.Textbox(
                            placeholder="输入问题或命令（/help 查看命令）...",
                            show_label=False,
                            scale=4,
                            lines=2
                        )
                        submit_btn = gr.Button("发送", variant="primary", scale=1)

                    # 控制按钮
                    with gr.Row():
                        clear_btn = gr.Button("清空对话", variant="secondary", size="sm")
                        stats_btn = gr.Button("查看统计", variant="secondary", size="sm")
                        help_btn = gr.Button("帮助", variant="secondary", size="sm")

                with gr.Column(scale=1):
                    # 模式控制区域
                    gr.Markdown("### 系统控制")

                    # 模式选择器
                    mode_radio = gr.Radio(
                        choices=["规则模式", "LLM模式", "混合模式"],
                        value="混合模式",
                        label="工作模式",
                        interactive=True
                    )

                    # 缓存控制
                    with gr.Row():
                        clear_cache_btn = gr.Button("清空缓存", size="sm")
                        test_classifier_btn = gr.Button("测试分类器", size="sm")

                    # 模式信息
                    gr.Markdown("""
                    **模式说明**:

                     **规则模式**: 基于关键词快速分类，速度快但准确性一般

                     **LLM模式**: 使用大语言模型智能分类，准确性高但速度较慢

                     **混合模式**: 结合规则和LLM，平衡速度和准确性
                    """)

                    # 快速示例
                    gr.Markdown("### 快速示例")

                    example_groups = [
                        ("多步骤任务", [
                            "先计算一下5+5等于多少，然后再介绍一下神经网络",
                            "第一步：计算25*4+18等于多少，第二步：解释什么是注意力机制",
                            "先总结Transformer的核心思想，再计算3的平方根",
                        ]),
                        ("单步任务", [
                            "神经网络的概念",
                            "计算一下100除以4等于多少",
                            "总结一下Transformer的核心思想",
                        ]),
                        ("分类测试", [
                            "你好",
                            "今天天气怎么样？",
                            "翻译'Hello world'成中文",
                            "什么是梯度下降算法？",
                        ])
                    ]

                    for group_name, examples in example_groups:
                        gr.Markdown(f"**{group_name}**")
                        for example in examples:
                            btn = gr.Button(
                                example[:25] + "..." if len(example) > 25 else example,
                                size="sm"
                            )
                            btn.click(lambda q=example: q, None, msg)

                    # 统计信息展示
                    stats_output = gr.Markdown("")

            # 事件绑定
            def update_mode(choice: str) -> str:
                mode_map = {"规则模式": "rule", "LLM模式": "llm", "混合模式": "hybrid"}
                mode = mode_map.get(choice, "hybrid")
                return self.switch_mode(mode)

            mode_radio.change(
                fn=update_mode,
                inputs=mode_radio,
                outputs=msg
            )

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

            stats_btn.click(
                fn=self.show_stats,
                inputs=None,
                outputs=stats_output
            )

            help_btn.click(
                fn=lambda: self.handle_command("/help"),
                inputs=None,
                outputs=stats_output
            )

            clear_cache_btn.click(
                fn=lambda: self.agent.clear_cache(),
                inputs=None,
                outputs=stats_output
            ).then(
                fn=lambda: " 缓存已清空",
                inputs=None,
                outputs=stats_output
            )

            test_classifier_btn.click(
                fn=lambda: self.handle_command("/test"),
                inputs=None,
                outputs=stats_output
            )

        return interface


# 十三、主函数
def main():
    print('=' * 50)
    print('智能知识库问答系统（支持LLM智能分类）')
    print('=' * 50)

    # 检查知识库
    if os.path.exists(Config.KNOWLEDGE_BASE_PATH):
        md_files = [f for f in os.listdir(Config.KNOWLEDGE_BASE_PATH) if f.endswith('.md')]
        print(f"发现 {len(md_files)} 个Markdown文件")
    else:
        print(f"知识库目录不存在")
        os.makedirs(Config.KNOWLEDGE_BASE_PATH, exist_ok=True)
        print(f"已创建知识库目录: {Config.KNOWLEDGE_BASE_PATH}")

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

    # 创建LLM分类器
    print("初始化LLM任务分类器...")
    llm_classifier = LLMTaskClassifier(
        llm_api=MiniMaxLLM.invoke,
        cache_size=Config.LLM_CLASSIFIER_CACHE_SIZE,
        use_cache=True
    )

    # 创建增强版Agent
    print("创建智能Agent...")
    agent = EnhancedIntelligentAgent(
        rag_chain=rag_chain,
        llm_classifier=llm_classifier
    )

    print('=' * 50)
    print('系统准备就绪！')
    print('支持的工作模式：')
    print('1. 规则模式（纯规则分类，速度快）')
    print('2. LLM模式（纯LLM分类，准确度高）')
    print('3. 混合模式（规则+LLM，平衡性能）')
    print('=' * 50)

    # 创建界面
    chat_interface = EnhancedChatInterface(agent, llm_classifier)
    interface = chat_interface.create_interface()

    print("请在浏览器中访问：http://localhost:7860")
    print("命令示例：")
    print("  /mode rule    - 切换到规则模式")
    print("  /mode llm     - 切换到LLM模式")
    print("  /mode hybrid  - 切换到混合模式")
    print("  /stats        - 查看系统统计")
    print("  /clear_cache  - 清空缓存")
    print("  /help         - 查看帮助信息")

    interface.launch(
        server_name="0.0.0.0",
        server_port=7860,
        share=False,
        debug=True
    )


if __name__ == "__main__":
    main()