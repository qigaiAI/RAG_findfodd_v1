"""
RAG系统配置文件
"""

from dataclasses import dataclass, field
from typing import Dict, Any

@dataclass
class RAGConfig:
    """RAG系统配置类"""

    # 路径配置
    # data_path: str = "../../data/C8/cook"
    data_path: str = "./dishes"
    index_save_path: str = "./vector_index"
    temp_index_save_path:str = "./temp_vector_index"

    # 模型配置
    embedding_model: str = "BAAI/bge-small-zh-v1.5"
    llm_model: str = "kimi-k2-0711-preview"

    # 检索配置
    top_k: int = 12

    # 生成配置
    temperature: float = 0.1
    max_tokens: int = 2048

    # 停用词
    GENERAL_STOP_WORDS: set = field(default_factory=lambda: {
        "我", "你", "他", "想", "吃", "做", "制作", "的", "了", "在", "是", "有",
        "要", "能", "会", "可以", "把", "用", "为", "之", "于", "及", "与", "等",
        "哦", "呢", "啊", "呀", "吧", "吗"
    })

    # 食谱领域停用词（通用领域词，无匹配价值）
    FOOD_STOP_WORDS: set = field(default_factory=lambda: {
        "做法", "步骤", "食材", "调料", "口感", "口味", "烹饪", "煮", "炖", "炒",
        "蒸", "炸", "烤", "焖", "烩", "凉拌", "煲汤", "小贴士", "注意事项", "准备"
    })
    
    @classmethod
    def from_dict(cls, config_dict: Dict[str, Any]) -> 'RAGConfig':
        """从字典创建配置对象"""
        return cls(**config_dict)
    
    def to_dict(self) -> Dict[str, Any]:
        """转换为字典"""
        return {
            'data_path': self.data_path,
            'index_save_path': self.index_save_path,
            'temp_index_save_path': self.temp_index_save_path,
            'embedding_model': self.embedding_model,
            'llm_model': self.llm_model,
            'top_k': self.top_k,
            'temperature': self.temperature,
            'max_tokens': self.max_tokens,
            'GENERAL_STOP_WORDS': self.GENERAL_STOP_WORDS,
            'FOOD_STOP_WORDS': self.FOOD_STOP_WORDS
        }

# 默认配置实例
DEFAULT_CONFIG = RAGConfig()
