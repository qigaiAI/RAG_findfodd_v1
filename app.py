"""
后端API服务
"""
import os
import json
from pathlib import Path
from flask import Flask, request, jsonify, CORS
from main import RecipeRAGSystem

app = Flask(__name__)
CORS(app)  # 启用CORS

# 初始化RAG系统
rag_system = None

# 分类映射
CATEGORY_MAPPING = {
    '荤菜': 'meat_dish',
    '素菜': 'vegetable_dish',
    '汤品': 'soup',
    '甜品': 'dessert',
    '早餐': 'breakfast',
    '主食': 'staple',
    '水产': 'aquatic',
    '调料': 'condiment',
    '饮品': 'drink'
}

# 难度映射
DIFFICULTY_STARS = {
    '非常简单': '★',
    '简单': '★★',
    '中等': '★★★',
    '困难': '★★★★',
    '非常困难': '★★★★★'
}

def init_rag_system():
    """初始化RAG系统"""
    global rag_system
    if rag_system is None:
        rag_system = RecipeRAGSystem()
        rag_system.initialize_system()
        rag_system.build_knowledge_base()

@app.route('/ask', methods=['POST'])
def ask():
    """处理用户问题"""
    try:
        data = request.json
        question = data.get('question', '')
        
        if not question:
            return jsonify({'error': '请输入问题'}), 400
        
        # 确保RAG系统已初始化
        if rag_system is None:
            init_rag_system()
        
        # 获取回答
        answer = ''
        for chunk in rag_system.ask_question(question, stream=True):
            answer += chunk
        
        return jsonify({'answer': answer})
    except Exception as e:
        print(f"处理问题时出错: {e}")
        return jsonify({'error': '处理问题时出错'}), 500

@app.route('/submit-dish', methods=['POST'])
def submit_dish():
    """接收用户提交的菜品"""
    try:
        data = request.json
        
        # 提取菜品信息
        dish_name = data.get('dish_name', '')
        category = data.get('category', '')
        difficulty = data.get('difficulty', '')
        ingredients = data.get('ingredients', '')
        steps = data.get('steps', '')
        
        if not all([dish_name, category, difficulty, ingredients, steps]):
            return jsonify({'error': '请填写完整的菜品信息'}), 400
        
        # 生成Markdown内容
        markdown_content = generate_markdown(dish_name, category, difficulty, ingredients, steps)
        
        # 保存到知识库
        save_dish_to_knowledge_base(dish_name, category, markdown_content)
        
        # 重新构建知识库
        init_rag_system()
        
        return jsonify({'success': True, 'message': '菜品提交成功'})
    except Exception as e:
        print(f"提交菜品时出错: {e}")
        return jsonify({'error': '提交菜品时出错'}), 500

def generate_markdown(dish_name, category, difficulty, ingredients, steps):
    """生成Markdown格式的菜品内容"""
    stars = DIFFICULTY_STARS.get(difficulty, '★★★')
    
    markdown = f"""
# {dish_name}

## 难度
{stars}

## 分类
{category}

## 食材
{ingredients}

## 步骤
{steps}
"""
    
    return markdown

def save_dish_to_knowledge_base(dish_name, category, content):
    """保存菜品到知识库"""
    # 获取分类对应的目录
    category_dir = CATEGORY_MAPPING.get(category, 'other')
    
    # 创建目录路径
    dish_path = Path('dishes') / category_dir / dish_name
    dish_path.mkdir(parents=True, exist_ok=True)
    
    # 保存Markdown文件
    md_file = dish_path / f"{dish_name}.md"
    with open(md_file, 'w', encoding='utf-8') as f:
        f.write(content)

if __name__ == '__main__':
    # 初始化RAG系统
    init_rag_system()
    
    # 启动服务器
    app.run(host='0.0.0.0', port=5000, debug=True)