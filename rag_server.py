"""
后端服务器，支持RAG系统调用和流式输出
"""

import http.server
import socketserver
import json
import os
import sys
import time
from pathlib import Path
from dotenv import load_dotenv

# 加载环境变量
load_dotenv()

PORT = 8001

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

# 尝试导入RAG系统
rag_system = None

class HTTPRequestHandler(http.server.SimpleHTTPRequestHandler):
    def do_POST(self):
        global rag_system
        if self.path == '/ask':
            # 处理用户问题
            content_length = int(self.headers['Content-Length'])
            post_data = self.rfile.read(content_length)
            data = json.loads(post_data.decode('utf-8'))
            question = data.get('question', '')
            stream = data.get('stream', False)
            
            if not question:
                self.send_response(400)
                self.send_header('Content-type', 'application/json')
                self.send_header('Access-Control-Allow-Origin', '*')
                self.end_headers()
                self.wfile.write(json.dumps({'error': '请输入问题'}).encode('utf-8'))
                return
            
            # 调用RAG系统生成回答或使用模拟回答
            if rag_system:
                try:
                    if stream:
                        # 流式返回
                        self.send_response(200)
                        self.send_header('Content-type', 'text/plain')
                        self.send_header('Access-Control-Allow-Origin', '*')
                        self.end_headers()
                        
                        # 流式生成回答
                        for chunk in rag_system.ask_question(question, stream=True):
                            self.wfile.write(chunk.encode('utf-8'))
                            self.wfile.flush()
                            time.sleep(0.03)  # 控制发送速度
                        return
                    else:
                        # 普通返回
                        answer = ''
                        for chunk in rag_system.ask_question(question, stream=True):
                            answer += chunk
                except Exception as e:
                    print(f"RAG系统回答失败: {e}")
                    # 使用模拟回答
                    answer = f"这是对问题 '{question}' 的回答。RAG系统回答失败: {e}"
            else:
                # 使用模拟回答
                answer = f"这是对问题 '{question}' 的回答。RAG系统未初始化，使用模拟回答。"
            
            if stream:
                # 流式返回模拟回答
                self.send_response(200)
                self.send_header('Content-type', 'text/plain')
                self.send_header('Access-Control-Allow-Origin', '*')
                self.end_headers()
                
                for char in answer:
                    self.wfile.write(char.encode('utf-8'))
                    self.wfile.flush()
                    time.sleep(0.03)  # 控制发送速度
            else:
                # 普通返回
                self.send_response(200)
                self.send_header('Content-type', 'application/json')
                self.send_header('Access-Control-Allow-Origin', '*')
                self.end_headers()
                self.wfile.write(json.dumps({'answer': answer}).encode('utf-8'))
        
        elif self.path == '/submit-dish':
            # 处理用户提交的菜品
            content_length = int(self.headers['Content-Length'])
            post_data = self.rfile.read(content_length)
            data = json.loads(post_data.decode('utf-8'))
            
            # 提取菜品信息
            dish_name = data.get('dish_name', '')
            category = data.get('category', '')
            difficulty = data.get('difficulty', '')
            ingredients = data.get('ingredients', '')
            steps = data.get('steps', '')
            
            if not all([dish_name, category, difficulty, ingredients, steps]):
                self.send_response(400)
                self.send_header('Content-type', 'application/json')
                self.send_header('Access-Control-Allow-Origin', '*')
                self.end_headers()
                self.wfile.write(json.dumps({'error': '请填写完整的菜品信息'}).encode('utf-8'))
                return
            
            # 生成Markdown内容
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
            
            # 保存到知识库
            category_dir = CATEGORY_MAPPING.get(category, 'other')
            dish_path = os.path.join('dishes', category_dir, dish_name)
            os.makedirs(dish_path, exist_ok=True)
            md_file = os.path.join(dish_path, f"{dish_name}.md")
            with open(md_file, 'w', encoding='utf-8') as f:
                f.write(markdown)
            
            # 重新初始化RAG系统
            try:
                from main import RecipeRAGSystem
                rag_system = RecipeRAGSystem()
                rag_system.initialize_system()
                rag_system.build_knowledge_base()
                print("RAG系统重新初始化成功！")
            except Exception as e:
                print(f"RAG系统重新初始化失败: {e}")
            
            self.send_response(200)
            self.send_header('Content-type', 'application/json')
            self.send_header('Access-Control-Allow-Origin', '*')
            self.end_headers()
            self.wfile.write(json.dumps({'success': True, 'message': '菜品提交成功'}).encode('utf-8'))
        
        elif self.path == '/delete-dish':
            # 处理用户删除的菜品
            content_length = int(self.headers['Content-Length'])
            post_data = self.rfile.read(content_length)
            data = json.loads(post_data.decode('utf-8'))
            
            # 提取菜品信息
            dish_name = data.get('dish_name', '')
            category = data.get('category', '')
            
            if not all([dish_name, category]):
                self.send_response(400)
                self.send_header('Content-type', 'application/json')
                self.send_header('Access-Control-Allow-Origin', '*')
                self.end_headers()
                self.wfile.write(json.dumps({'error': '请填写完整的菜品信息'}).encode('utf-8'))
                return
            
            # 从知识库中删除菜品
            category_dir = CATEGORY_MAPPING.get(category, 'other')
            dish_path = os.path.join('dishes', category_dir, dish_name)
            md_file = os.path.join(dish_path, f"{dish_name}.md")
            
            success = False
            if os.path.exists(md_file):
                os.remove(md_file)
                if not os.listdir(dish_path):
                    os.rmdir(dish_path)
                success = True
            
            if success:
                # 重新初始化RAG系统
                try:
                    from main import RecipeRAGSystem
                    rag_system = RecipeRAGSystem()
                    rag_system.initialize_system()
                    rag_system.build_knowledge_base()
                    print("RAG系统重新初始化成功！")
                except Exception as e:
                    print(f"RAG系统重新初始化失败: {e}")
                
                self.send_response(200)
                self.send_header('Content-type', 'application/json')
                self.send_header('Access-Control-Allow-Origin', '*')
                self.end_headers()
                self.wfile.write(json.dumps({'success': True, 'message': '菜品删除成功'}).encode('utf-8'))
            else:
                self.send_response(404)
                self.send_header('Content-type', 'application/json')
                self.send_header('Access-Control-Allow-Origin', '*')
                self.end_headers()
                self.wfile.write(json.dumps({'error': '菜品不存在'}).encode('utf-8'))
        
        else:
            self.send_response(404)
            self.end_headers()
            self.wfile.write(b'Not Found')
    
    def do_OPTIONS(self):
        # 处理CORS预检请求
        self.send_response(200)
        self.send_header('Access-Control-Allow-Origin', '*')
        self.send_header('Access-Control-Allow-Methods', 'POST, OPTIONS')
        self.send_header('Access-Control-Allow-Headers', 'Content-Type')
        self.end_headers()

def main():
    print('=== 启动后端服务器 ===')
    print('当前工作目录:', os.getcwd())
    print('Python版本:', sys.version)
    print('MOONSHOT_API_KEY:', os.getenv('MOONSHOT_API_KEY') is not None)
    
    # 确保dishes目录存在
    dishes_dir = Path('./dishes')
    if not dishes_dir.exists():
        print("创建dishes目录...")
        dishes_dir.mkdir(parents=True, exist_ok=True)
        print("dishes目录创建成功！")
    
    # 尝试初始化RAG系统
    global rag_system
    print('尝试初始化RAG系统...')
    try:
        print('导入RecipeRAGSystem...')
        from main import RecipeRAGSystem
        print('创建RecipeRAGSystem实例...')
        rag_system = RecipeRAGSystem()
        print('初始化系统...')
        rag_system.initialize_system()
        print('构建知识库...')
        rag_system.build_knowledge_base()
        print("RAG系统初始化成功！")
    except Exception as e:
        print(f"RAG系统初始化失败: {e}")
        import traceback
        traceback.print_exc()
        print("使用模拟回答模式...")
        rag_system = None
    
    print(f'启动服务器在 http://localhost:{PORT}')
    try:
        with socketserver.TCPServer(('', PORT), HTTPRequestHandler) as httpd:
            print(f'Server running at http://localhost:{PORT}')
            print('服务器已启动，等待请求...')
            httpd.serve_forever()
    except Exception as e:
        print(f'服务器启动失败: {e}')
        import traceback
        traceback.print_exc()

if __name__ == '__main__':
    main()
