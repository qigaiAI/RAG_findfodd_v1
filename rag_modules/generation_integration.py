"""
生成集成模块
"""

import os
import logging
from typing import List

from langchain_core.prompts import ChatPromptTemplate, PromptTemplate
from langchain_community.chat_models.moonshot import MoonshotChat
from langchain_core.documents import Document
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser

logger = logging.getLogger(__name__)

class GenerationIntegrationModule:
    """生成集成模块 - 负责LLM集成和回答生成"""
    
    def __init__(self, model_name: str = "kimi-k2-0711-preview", temperature: float = 0.1, max_tokens: int = 2048):
        """
        初始化生成集成模块
        
        Args:
            model_name: 模型名称
            temperature: 生成温度
            max_tokens: 最大token数
        """
        self.model_name = model_name
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.llm = None
        self.setup_llm()
    
    def setup_llm(self):
        """初始化大语言模型"""
        logger.info(f"正在初始化LLM: {self.model_name}")

        api_key = os.getenv("MOONSHOT_API_KEY")
        if not api_key:
            raise ValueError("请设置 MOONSHOT_API_KEY 环境变量")

        self.llm = MoonshotChat(
            model=self.model_name,
            temperature=self.temperature,
            max_tokens=self.max_tokens,
            moonshot_api_key=api_key
        )
        
        logger.info("LLM初始化完成")
    
    def generate_basic_answer(self, query: str, context_docs: List[Document]) -> str:
        """
        生成基础回答

        Args:
            query: 用户查询
            context_docs: 上下文文档列表

        Returns:
            生成的回答
        """
        context = self._build_context(context_docs)

        prompt = ChatPromptTemplate.from_template("""
你是一位专业的烹饪助手。请根据以下食谱信息回答用户的问题。

用户问题: {question}

相关食谱信息:
{context}

请提供详细、实用的回答。如果信息不足，请诚实说明。

回答:""")

        # 使用LCEL构建链
        chain = (
            {"question": RunnablePassthrough(), "context": lambda _: context}
            | prompt
            | self.llm
            | StrOutputParser()
        )

        response = chain.invoke(query)
        return response
    
    def generate_step_by_step_answer(self, query: str, context_docs: List[Document]) -> str:
        """
        生成分步骤回答

        Args:
            query: 用户查询
            context_docs: 上下文文档列表

        Returns:
            分步骤的详细回答
        """
        context = self._build_context(context_docs)

        prompt = ChatPromptTemplate.from_template("""
你是一位专业的烹饪导师。请根据食谱信息，为用户提供详细的分步骤指导。

用户问题: {question}

相关食谱信息:
{context}

请灵活组织回答，建议包含以下部分（可根据实际内容调整）：

## 🥘 菜品介绍
[简要介绍菜品特点和难度]

## 🛒 所需食材
[列出主要食材和用量]

## 👨‍🍳 制作步骤
[详细的分步骤说明，每步包含具体操作和大概所需时间]

## 💡 制作技巧
[仅在有实用技巧时包含。优先使用原文中的实用技巧，如果原文的"附加内容"与烹饪无关或为空，可以基于制作步骤总结关键要点，或者完全省略此部分]

注意：
- 根据实际内容灵活调整结构
- 不要强行填充无关内容或重复制作步骤中的信息
- 重点突出实用性和可操作性
- 如果没有额外的技巧要分享，可以省略制作技巧部分

回答:""")

        chain = (
            {"question": RunnablePassthrough(), "context": lambda _: context}
            | prompt
            | self.llm
            | StrOutputParser()
        )

        response = chain.invoke(query)
        return response
    
    def query_rewrite(self, query: str) -> str:
        """
        智能查询重写 - 让大模型判断是否需要重写查询

        Args:
            query: 原始查询

        Returns:
            重写后的查询或原查询
        """
        prompt = PromptTemplate(
            template="""你是一个专业的智能查询分析助手，核心职责是分析用户的食谱搜索原始查询，判断是否需要重写以提升搜索效果，重写后的内容必须按要求包含菜量、烹饪难度、菜品类型相关信息，未明确的信息需结合查询原意合理补充适配的通用内容。

分析规则：
具体明确的查询（直接返回原查询，无需额外补充信息）
包含具体菜品名称：如 "宫保鸡丁怎么做"、"红烧肉的制作方法"
明确的制作询问：如 "蛋炒饭需要什么食材"、"糖醋排骨的步骤"
具体的烹饪技巧：如 "如何炒菜不粘锅"、"怎样调制糖醋汁"
模糊不清的查询（必须按规则重写，且补充菜量、难度、类型信息）
过于宽泛：如 "做菜"、"有什么好吃的"、"推荐个菜"
缺乏具体信息：如 "川菜"、"素菜"、"简单的"
口语化表达：如 "想吃点什么"、"有饮品推荐吗"
食物类别不清晰：如 "不要肉的菜"（需明确为素菜并补充其他信息）
场合性查询：涉及特定场合但未明确菜品类型，如 "过生日的菜"、"请客吃饭"、"过节吃什么"（需引入 “大菜”、“硬菜”、“宴席”、“精致” 等关键词）

核心重写原则
严格保持用户查询的原意不变，仅做补充和明确化；
增加专业的烹饪相关术语，贴合食谱搜索场景；
明确菜谱的具体类别，无明确类别时结合原意合理选择；
明确菜品的特点、食用需求或场合属性；
重写后内容必须包含以下三类信息，缺一不可：
菜量：适配日常场景的通用菜量，如「2-3 人份」「4-5 人份」「一人食」；
烹饪难度：必须从固定值中选择，可选值为「非常简单」「简单」「中等」「困难」「非常困难」；
菜品类型：必须从固定值中选择，可选值为「荤菜」「素菜」「汤品」「甜品」「早餐」「主食」「水产」「调料」「饮品」；
菜品口味：必须从固定值中选择，可选值为 [清淡] [微辣]  [麻辣]  [酸甜]  [咸鲜]  [香辣]  [甜口] [其他]，如无法明确口味可选「其他」；
场合性查询需优先引入 “大菜”“硬菜”“宴席”“精致”“高档” 等关键词，匹配高端或主菜类菜谱。

格式要求
直接返回重写后的查询内容，无需额外分析、解释，仅输出最终重写结果即可。

参考示例
原始查询：做菜 → 重写结果：2-3 人份简单难度素菜的家常菜谱制作方法
原始查询：有饮品推荐吗 → 重写结果：2-3 人份非常简单难度饮品的家常制作方法推荐
原始查询：推荐个菜 → 重写结果：一人食简单难度荤菜的家常菜推荐
原始查询：川菜 → 重写结果：4-5 人份中等难度荤菜的经典川菜菜谱制作方法
原始查询：过生日的菜 → 重写结果：4-5 人份中等难度荤菜的生日宴席精致大菜推荐
原始查询：请客吃饭的菜 → 重写结果：5-6 人份中等难度荤菜的宴客高档硬菜菜谱推荐
原始查询：过节吃什么 → 重写结果：4-5 人份中等难度荤菜的节日宴席精致菜谱推荐
原始查询：不要肉的菜 → 重写结果：2-3 人份简单难度素菜的家常菜谱制作方法
原始查询：宫保鸡丁怎么做 → 重写结果：宫保鸡丁怎么做
原始查询：红烧肉需要什么食材 → 重写结果：红烧肉需要什么食材
原始查询：简单的早餐 → 重写结果：一人食非常简单难度早餐的家常制作方法推荐

待分析原始查询
{query}
""",
            input_variables=["query"]
        )

        chain = (
            {"query": RunnablePassthrough()}
            | prompt
            | self.llm
            | StrOutputParser()
        )

        response = chain.invoke(query).strip()#移除字符串开头和结尾的指定字符（默认移除空白字符）

        # 记录重写结果
        if response != query:
            logger.info(f"查询已重写: '{query}' → '{response}'")
        else:
            logger.info(f"查询无需重写: '{query}'")

        return response



    def query_router(self, query: str) -> str:
        """
        查询路由 - 根据查询类型选择不同的处理方式

        Args:
            query: 用户查询

        Returns:
            路由类型 ('list', 'detail', 'general')
        """
        prompt = ChatPromptTemplate.from_template("""
根据用户的问题，将其分类为以下三种类型之一：

1. 'list' - 用户想要获取菜品列表或推荐，只需要菜名和推荐理由
   例如：推荐几个素菜、有什么川菜、给我3个简单的菜

2. 'detail' - 用户想要具体的制作方法或详细信息
   例如：宫保鸡丁怎么做、制作步骤、需要什么食材

3. 'general' - 一般性问题,和做饭相关但不属于上述两类的查询，可能需要提供综合性的回答
   例如：什么是川菜、制作技巧、营养价值
   
4. 'other' - 其他与烹饪不相关问题
    例如：今天天气怎么样、你是谁、讲个笑话   


请只返回分类结果：list、detail 、general 或 other，不要任何解释或多余的文本。

用户问题: {query}

分类结果:""")

        chain = (
            {"query": RunnablePassthrough()}
            | prompt
            | self.llm
            | StrOutputParser()
        )

        result = chain.invoke(query).strip().lower()  #strip()移除字符串开头和结尾的指定字符（默认移除空白字符），lower()将字符串转换为小写

        # 确保返回有效的路由类型
        if result in ['list', 'detail', 'general', 'other']:
            return result
        else:
            return 'general'  # 默认类型

    '''
  def generate_list_answer(self, query: str, context_docs: List[Document]) -> str:
        """
        生成列表式回答 - 适用于推荐类查询

        Args:
            query: 用户查询
            context_docs: 上下文文档列表

        Returns:
            列表式回答
        """
        if not context_docs:
            return "抱歉，没有找到相关的菜品信息。"

        # 提取菜品名称
        dish_names = []
        for doc in context_docs:
            dish_name = doc.metadata.get('dish_name', '未知菜品')
            if dish_name not in dish_names:
                dish_names.append(dish_name)

        # 构建简洁的列表回答
        if len(dish_names) == 1:
            return f"为您推荐：{dish_names[0]}"
        elif len(dish_names) <= 3:
            return f"为您推荐以下菜品：\n" + "\n".join([f"{i+1}. {name}" for i, name in enumerate(dish_names)])
        else:
            return f"为您推荐以下菜品：\n" + "\n".join([f"{i+1}. {name}" for i, name in enumerate(dish_names[:3])]) + f"\n\n还有其他 {len(dish_names)-3} 道菜品可供选择。"

'''


    def generate_list_answer_stream(self, query: str, context_docs: List[Document]) -> str:
        context = self._build_context(context_docs)

        prompt = ChatPromptTemplate.from_template("""
        你是一位专业且贴心的烹饪助手。你的任务是根据用户的问题和提供的食谱信息，向用户推荐合适的菜品。

### 核心原则：
1.  **只基于提供的食谱信息**：请严格依据下面【相关食谱信息】中的内容进行推荐，不要使用你自己的外部知识补充食谱细节。
2.  **以用户为中心**：仔细分析用户的问题，理解他们的潜在需求（例如：口味偏好、用餐人群、特殊场合、烹饪时间限制等）。
3.  **诚实透明**：如果提供的食谱信息中没有合适的菜品，或者信息不足以支撑推荐，请如实告知用户，并引导他们如何获取更精准的帮助。

### 用户问题：
{query}

### 相关食谱信息：
{context}

请根据以上信息，完成以下任务：

1.  **分析用户需求**：简要分析用户问题中可能隐含的饮食需求（例如：用户说“想吃辣的”，可能偏好川湘菜；用户说“给孩子做”，可能需要简单、营养的菜品）。
2.  **筛选合适菜品**：从【相关食谱信息】中挑选出最匹配用户需求和问题场景的菜品。
3.  **给出推荐列表**：对于每道推荐的菜品，请按以下格式提供信息：
    -   **菜品名称**：
    -   **推荐原因**：结合食谱信息和用户需求进行说明。例如：
        -   **口味特点**：是否酸辣、香甜、清淡等，是否满足用户口味。
        -   **制作难度**：新手友好/需要一定厨艺，是否符合用户时间或能力。
        -   **适合场景**：是否适合日常、聚会、节日、便当等。
        -   **营养亮点**：如果食谱中有特别营养的食材或搭配，可以提及。
        -   **与用户需求的契合点**：明确指出这道菜为什么能解决用户的问题。
4.  **处理信息不足的情况**：
    -   如果【相关食谱信息】为空，或者其中没有与用户问题相关的菜品，请礼貌地表示：“根据目前提供的食谱，暂时没有找到特别合适的推荐。您可以尝试描述更具体的需求（比如想吃什么菜系、有什么食材），我来为您精准查找。”
    -   如果信息中只有部分相关（比如用户想要鸡肉食谱，但只有鱼香肉丝），可以诚实说明，并询问是否愿意尝试其他类似菜品。

### 推荐列表：
（请在此处列出你的推荐）：
        """)
        chain = (
                {"query": RunnablePassthrough(), "context": lambda _: context}
                | prompt
                | self.llm
                | StrOutputParser()
        )

        for chunk in chain.stream(query):
            yield chunk

    def generate_basic_answer_stream(self, query: str, context_docs: List[Document]):
        """
        生成基础回答 - 流式输出

        Args:
            query: 用户查询
            context_docs: 上下文文档列表

        Yields:
            生成的回答片段
        """
        context = self._build_context(context_docs)

        prompt = ChatPromptTemplate.from_template("""
你是一位专业的烹饪助手。请根据以下食谱信息回答用户的问题。

用户问题: {question}

相关食谱信息:
{context}

请提供详细、实用的回答。如果信息不足，请诚实说明。

回答:""")

        chain = (
            {"question": RunnablePassthrough(), "context": lambda _: context}
            | prompt
            | self.llm
            | StrOutputParser()
        )

        for chunk in chain.stream(query):
            yield chunk

    def generate_step_by_step_answer_stream(self, query: str, context_docs: List[Document]):
        """
        生成详细步骤回答 - 流式输出

        Args:
            query: 用户查询
            context_docs: 上下文文档列表

        Yields:
            详细步骤回答片段
        """
        context = self._build_context(context_docs)

        prompt = ChatPromptTemplate.from_template("""
你是一位专业的烹饪导师。请根据食谱信息，为用户提供详细的分步骤指导。

用户问题: {question}

相关食谱信息:
{context}

请灵活组织回答，建议包含以下部分（可根据实际内容调整）：

## 🥘 菜品介绍
[简要介绍菜品特点和难度]

## 🛒 所需食材
[列出主要食材和用量]

## 👨‍🍳 制作步骤
[详细的分步骤说明，每步包含具体操作和大概所需时间]

## 💡 制作技巧
[仅在有实用技巧时包含。如果原文的"附加内容"与烹饪无关或为空，可以基于制作步骤总结关键要点，或者完全省略此部分]

注意：
- 根据实际内容灵活调整结构
- 不要强行填充无关内容
- 重点突出实用性和可操作性

回答:""")

        chain = (
            {"question": RunnablePassthrough(), "context": lambda _: context}
            | prompt
            | self.llm
            | StrOutputParser()
        )

        for chunk in chain.stream(query):
            yield chunk

    def _build_context(self, docs: List[Document], max_length: int = 2000) -> str:
        """
        构建上下文字符串
        
        Args:
            docs: 文档列表
            max_length: 最大长度
            
        Returns:
            格式化的上下文字符串
        """
        if not docs:
            return "暂无相关食谱信息。"
        
        context_parts = []
        current_length = 0
        
        for i, doc in enumerate(docs, 1):
            # 添加元数据信息
            metadata_info = f"【食谱 {i}】"
            if 'dish_name' in doc.metadata:
                metadata_info += f" {doc.metadata['dish_name']}"
            if 'category' in doc.metadata:
                metadata_info += f" | 分类: {doc.metadata['category']}"
            if 'difficulty' in doc.metadata:
                metadata_info += f" | 难度: {doc.metadata['difficulty']}"
            
            # 构建文档文本
            doc_text = f"{metadata_info}\n{doc.page_content}\n"
            
            # 检查长度限制
            if current_length + len(doc_text) > max_length:
                break
            
            context_parts.append(doc_text)
            current_length += len(doc_text)
        
        return "\n" + "="*50 + "\n".join(context_parts)
