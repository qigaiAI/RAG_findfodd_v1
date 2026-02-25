"""
RAG系统主程序
"""

import os
import sys
import logging
from pathlib import Path
from typing import List

# 添加模块路径
sys.path.append(str(Path(__file__).parent))

from dotenv import load_dotenv
from config import DEFAULT_CONFIG, RAGConfig
from rag_modules import (
    DataPreparationModule,
    IndexConstructionModule,
    RetrievalOptimizationModule,
    GenerationIntegrationModule
)

# 加载环境变量
load_dotenv()

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class RecipeRAGSystem:
    """食谱RAG系统主类"""

    def __init__(self, config: RAGConfig = None):
        """
        初始化RAG系统

        Args:
            config: RAG系统配置，默认使用DEFAULT_CONFIG
        """
        self.config = config or DEFAULT_CONFIG
        self.data_module = None
        self.index_module = None
        self.retrieval_module = None
        self.generation_module = None

        # 检查数据路径
        if not Path(self.config.data_path).exists():
            raise FileNotFoundError(f"数据路径不存在: {self.config.data_path}")

        # 检查API密钥
        if not os.getenv("MOONSHOT_API_KEY"):
            raise ValueError("请设置 MOONSHOT_API_KEY 环境变量")
    
    def initialize_system(self):
        """初始化所有模块"""
        print("🚀 正在初始化RAG系统...")

        # 1. 初始化数据准备模块
        print("初始化数据准备模块...")
        self.data_module = DataPreparationModule(self.config.data_path)

        # 2. 初始化索引构建模块
        print("初始化索引构建模块...")
        self.index_module = IndexConstructionModule(
            model_name=self.config.embedding_model,
            index_save_path=self.config.index_save_path
        )
#为什么没有检索优化模块的初始化
        # 3. 初始化生成集成模块
        print("🤖 初始化生成集成模块...")
        self.generation_module = GenerationIntegrationModule(
            model_name=self.config.llm_model,
            temperature=self.config.temperature,
            max_tokens=self.config.max_tokens
        )

        print("✅ 系统初始化完成！")
    
    def build_knowledge_base(self):
        """构建知识库"""
        print("\n正在构建知识库...")

        # 1. 尝试加载已保存的索引
        vectorstore = self.index_module.load_index()

        if vectorstore is not None:
            print("✅ 成功加载已保存的向量索引！")
            # 仍需要加载文档和分块用于检索模块
            print("加载食谱文档...")
            self.data_module.load_documents()
            print("进行文本分块...")
            chunks = self.data_module.chunk_documents()
        else:
            print("未找到已保存的索引，开始构建新索引...")

            # 2. 加载文档
            print("加载食谱文档...")
            self.data_module.load_documents()

            # 3. 文本分块
            print("进行文本分块...")
            chunks = self.data_module.chunk_documents()

            # 4. 构建向量索引
            print("构建向量索引...")
            vectorstore = self.index_module.build_vector_index(chunks)

            # 5. 保存索引
            print("保存向量索引...")
            self.index_module.save_index()

        # 6. 初始化检索优化模块
        print("初始化检索优化...")
        self.retrieval_module = RetrievalOptimizationModule(vectorstore, chunks)

        # 7. 显示统计信息
        stats = self.data_module.get_statistics()
        print(f"\n📊 知识库统计:")
        print(f"   文档总数: {stats['total_documents']}")
        print(f"   文本块数: {stats['total_chunks']}")
        print(f"   菜品分类: {list(stats['categories'].keys())}")
        print(f"   难度分布: {stats['difficulties']}")

        print("✅ 知识库构建完成！")
    
    def ask_question(self, question: str, stream: bool = False):
        """
        回答用户问题

        Args:
            question: 用户问题
            stream: 是否使用流式输出

        Returns:
            生成的回答或生成器
        """
        if not all([self.retrieval_module, self.generation_module]):
            raise ValueError("请先构建知识库")
        
        print(f"\n❓ 用户问题: {question}")

        # 1. 查询路由
        route_type = self.generation_module.query_router(question)
        print(f"🎯 查询类型: {route_type}")

        # 2. 智能查询重写（根据路由类型）
        if route_type == 'other':
            # 其他查询直接返回不进行检索
            rewritten_query = question
            print(f"📝 列表查询保持原样: {question}")
        else:
            # 详细查询和一般查询使用智能重写
            print("🤖 智能分析查询...")
            rewritten_query = self.generation_module.query_rewrite(question)

            # 3. 检索相关子块（自动应用元数据过滤）

            print("🔍 检索相关文档...")
            filters = self._extract_filters_from_query(rewritten_query)
            if filters:
                print(f"应用过滤条件: {filters}")
                # 元数据过滤，
                filters_chunks = self.retrieval_module.metadata_filtered_search(rewritten_query, filters,
                                                                                 top_k=self.config.top_k*3)
                relevant_chunks = self.retrieval_module.hybrid_search_from_filters(rewritten_query,filters_chunks, top_k=self.config.top_k)
            else:
                relevant_chunks = self.retrieval_module.hybrid_search(rewritten_query, top_k=self.config.top_k)
            # 这个是一个示例，展示了如何从用户查询中提取元数据过滤条件（如分类和难度），并将这些条件应用于检索过程，以提高检索结果的相关性和准确性。您可以根据需要调整过滤条件的提取逻辑和应用方式。

            # relevant_chunks = self.retrieval_module.hybrid_search(rewritten_query, top_k=self.config.top_k)

            # 显示检索到的子块信息
            if relevant_chunks:
                chunk_info = []  # 这句话是为了在输出中显示每个相关文档块的菜品名称和章节标题（如果有的话），以便更清晰地了解检索结果的内容和来源
                for chunk in relevant_chunks:
                    dish_name = chunk.metadata.get('dish_name', '未知菜品')
                    # 尝试从内容中提取章节标题
                    content_preview = chunk.page_content[:100].strip()
                    if content_preview.startswith('#'):
                        # 如果是标题开头，提取标题（仅取第一行）
                        title_end = content_preview.find('\n') if '\n' in content_preview else len(content_preview)
                        section_title = content_preview[:title_end].replace('#', '').strip()
                        chunk_info.append(f"{dish_name}({section_title})")
                    else:
                        chunk_info.append(f"{dish_name}(内容片段)")

                print(f"找到 {len(relevant_chunks)} 个相关文档块: {', '.join(chunk_info)}")
            else:
                print(f"找到 {len(relevant_chunks)} 个相关文档块")

            # 4. 检查是否找到相关内容
            if not relevant_chunks:
                return "抱歉，没有找到相关的食谱信息。请尝试其他菜品名称或关键词。"



        # 5. 根据路由类型选择回答方式
        if route_type == 'list':
            # 列表查询：返回菜品名称和简要介绍推荐理由
            print("📋 生成菜品列表...")
            relevant_docs = self.data_module.get_parent_documents(relevant_chunks)

            # 显示找到的文档名称
            doc_names = []
            for doc in relevant_docs:
                dish_name = doc.metadata.get('dish_name', '未知菜品')
                doc_names.append(dish_name)

            if doc_names:
                print(f"找到文档: {', '.join(doc_names)}")

            return self.generation_module.generate_list_answer_stream(question, relevant_docs)
        if route_type == 'other':
            # 其他查询直接返回不进行检索
            print("❌ 该问题与烹饪无关，直接生成回答...")
            return self.generation_module.generate_basic_answer(question, [])
        else:
            # 详细查询：获取完整文档并生成详细回答
            print("获取完整文档...")
            relevant_docs = self.data_module.get_parent_documents(relevant_chunks)

            # 显示找到的文档名称
            doc_names = []
            for doc in relevant_docs:
                dish_name = doc.metadata.get('dish_name', '未知菜品')
                doc_names.append(dish_name)

            if doc_names:
                print(f"找到文档: {', '.join(doc_names)}")
            else:
                print(f"对应 {len(relevant_docs)} 个完整文档")

            print("✍️ 生成详细回答...")

            # 根据路由类型自动选择回答模式
            if route_type == "detail":
                # 详细查询使用分步指导模式
                if stream:
                    return self.generation_module.generate_step_by_step_answer_stream(question, relevant_docs)
                else:
                    return self.generation_module.generate_step_by_step_answer(question, relevant_docs)
            else:
                # 一般查询使用基础回答模式
                if stream:
                    return self.generation_module.generate_basic_answer_stream(question, relevant_docs)
                else:
                    return self.generation_module.generate_basic_answer(question, relevant_docs)
    
    def _extract_filters_from_query(self, query: str) -> dict:
        """
        从重写的用户问题中提取元数据过滤条件
        """
        #提取关键词太有限，可以考虑使用更复杂的NLP方法来识别用户意图中的分类和难度信息
        filters = {}
        # 分类关键词
        category_keywords = DataPreparationModule.get_supported_categories()
        for cat in category_keywords:
            if cat in query:
                filters['category'] = cat
                break

        # 难度关键词
        difficulty_keywords = DataPreparationModule.get_supported_difficulties()
        for diff in sorted(difficulty_keywords, key=len, reverse=True):
            if diff in query:
                filters['difficulty'] = diff
                break

        return filters

    '''
    def search_by_category(self, category: str, query: str = "") -> List[str]:
        """
        按分类搜索菜品
        
        Args:
            category: 菜品分类
            query: 可选的额外查询条件
            
        Returns:
            菜品名称列表
        """
        if not self.retrieval_module:
            raise ValueError("请先构建知识库")
        
        # 使用元数据过滤搜索
        search_query = query if query else category
        filters = {"category": category}
        
        docs = self.retrieval_module.metadata_filtered_search(search_query, filters, top_k=10)
        
        # 提取菜品名称
        dish_names = []
        for doc in docs:
            dish_name = doc.metadata.get('dish_name', '未知菜品')
            if dish_name not in dish_names:
                dish_names.append(dish_name)
        
        return dish_names
        '''
      #这个方法是一个示例，展示了如何通过分类进行搜索。它使用了元数据过滤功能来限制搜索结果仅包含指定分类的菜品，并返回这些菜品的名称列表。您可以根据需要调整搜索条件和返回的信息。

    '''
    def get_ingredients_list(self, dish_name: str) -> str:
        """
        获取指定菜品的食材信息

        Args:
            dish_name: 菜品名称

        Returns:
            食材信息
        """
        if not all([self.retrieval_module, self.generation_module]):
            raise ValueError("请先构建知识库")

        # 搜索相关文档
        docs = self.retrieval_module.hybrid_search(dish_name, top_k=3)

        # 生成食材信息
        answer = self.generation_module.generate_basic_answer(f"{dish_name}需要什么食材？", docs)

        return answer
    '''
    def run_interactive(self):
        """运行交互式问答"""
        print("=" * 60)
        print("🍽️📕 尝尝咸淡RAG系统 - 交互式问答  🍽️")
        print("=" * 60)
        print("💡 解决您的选择困难症，告别'今天吃什么'的世纪难题！")
        
        # 初始化系统
        self.initialize_system()
        
        # 构建知识库
        self.build_knowledge_base()
        
        print("\n交互式问答 (输入'退出'结束):")
        
        while True:
            try:
                user_input = input("\n您的问题: ").strip()
                if user_input.lower() in ['退出', 'quit', 'exit', '']:
                    break

                '''
                # 询问是否使用流式输出
                stream_choice = input("是否使用流式输出? (y/n, 默认y): ").strip().lower()
                use_stream = stream_choice != 'n'
                
                
                print("\n回答:")
                if use_stream:
                    # 流式输出
                    for chunk in self.ask_question(user_input, stream=True):
                        print(chunk, end="", flush=True)
                    print("\n")
                else:
                    # 普通输出
                    answer = self.ask_question(user_input, stream=False)
                    print(f"{answer}\n")
                '''#这个代码块是一个示例，展示了如何在交互式问答中让用户选择是否使用流式输出。根据用户的选择，系统会调用不同的生成方法来返回回答，并以相应的方式显示结果。您可以根据需要调整用户交互的细节和输出格式。
                for chunk in self.ask_question(user_input, stream=True):
                    print(chunk, end="", flush=True)
                print("\n")

            except KeyboardInterrupt:
                break
            except Exception as e:
                print(f"处理问题时出错: {e}")
        
        print("\n感谢使用尝尝咸淡RAG系统！")



def main():
    """主函数"""
    try:
        # 创建RAG系统
        rag_system = RecipeRAGSystem()
        
        # 运行交互式问答
        rag_system.run_interactive()
        
    except Exception as e:
        logger.error(f"系统运行出错: {e}")
        print(f"系统错误: {e}")

if __name__ == "__main__":
    main()
