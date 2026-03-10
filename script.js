// 前端脚本

// API基础URL
const API_BASE_URL = '';

// DOM元素
const chatMessages = document.getElementById('chat-messages');
const userInput = document.getElementById('user-input');
const sendButton = document.getElementById('send-button');
const submitForm = document.getElementById('submit-form');
const deleteForm = document.getElementById('delete-form');

// 初始化
function init() {
    // 绑定事件
    sendButton.addEventListener('click', sendMessage);
    userInput.addEventListener('keypress', (e) => {
        if (e.key === 'Enter') {
            sendMessage();
        }
    });
    
    submitForm.addEventListener('submit', handleSubmitDish);
    deleteForm.addEventListener('submit', handleDeleteDish);
}

// 发送消息
async function sendMessage() {
    const message = userInput.value.trim();
    if (!message) return;
    
    // 添加用户消息到对话
    addMessage('user', message);
    userInput.value = '';
    
    // 移除加载状态，直接显示空的机器人消息框
    const botMessageElement = addMessage('bot', '');
    
    try {
        // 调用API获取回答（流式）
        const response = await fetch(`${API_BASE_URL}/ask`, {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json'
            },
            body: JSON.stringify({ question: message, stream: true })
        });
        
        if (!response.ok) {
            throw new Error('API请求失败');
        }
        
        // 处理流式响应
        const reader = response.body.getReader();
        const decoder = new TextDecoder();
        let accumulatedAnswer = '';
        
        while (true) {
            const { done, value } = await reader.read();
            if (done) break;
            
            // 解码并添加到回答中
            const chunk = decoder.decode(value, { stream: true });
            accumulatedAnswer += chunk;
            
            // 更新消息内容
            updateMessageContent(botMessageElement, accumulatedAnswer);
        }
        
        // 完成后确保所有内容都被解码
        const finalChunk = decoder.decode();
        if (finalChunk) {
            accumulatedAnswer += finalChunk;
            updateMessageContent(botMessageElement, accumulatedAnswer);
        }
    } catch (error) {
        console.error('发送消息失败:', error);
        // 更新消息内容为错误信息
        updateMessageContent(botMessageElement, '抱歉，系统暂时无法回答你的问题，请稍后再试。');
    }
}

// 添加消息到对话
function addMessage(type, content) {
    const messageDiv = document.createElement('div');
    messageDiv.className = `message ${type}-message`;
    
    const messageContent = document.createElement('div');
    messageContent.className = 'message-content';
    messageContent.innerHTML = `<p>${content}</p>`;
    
    messageDiv.appendChild(messageContent);
    chatMessages.appendChild(messageDiv);
    
    // 滚动到底部
    chatMessages.scrollTop = chatMessages.scrollHeight;
    
    return messageDiv;
}

// 更新消息内容
function updateMessageContent(messageElement, content) {
    const messageContent = messageElement.querySelector('.message-content');
    if (messageContent) {
        messageContent.innerHTML = `<p>${content}</p>`;
        // 滚动到底部
        chatMessages.scrollTop = chatMessages.scrollHeight;
    }
}

// 添加加载消息
function addLoadingMessage() {
    const messageDiv = document.createElement('div');
    messageDiv.className = 'message bot-message';
    messageDiv.id = 'loading-message';
    
    const messageContent = document.createElement('div');
    messageContent.className = 'message-content';
    messageContent.innerHTML = '<div class="loading"></div>';
    
    messageDiv.appendChild(messageContent);
    chatMessages.appendChild(messageDiv);
    
    // 滚动到底部
    chatMessages.scrollTop = chatMessages.scrollHeight;
    
    return 'loading-message';
}

// 移除加载消息
function removeLoadingMessage(id) {
    const loadingMessage = document.getElementById(id);
    if (loadingMessage) {
        loadingMessage.remove();
    }
}

// 处理提交菜品
async function handleSubmitDish(e) {
    e.preventDefault();
    
    // 获取表单数据
    const dishName = document.getElementById('dish-name').value;
    const category = document.getElementById('dish-category').value;
    const difficulty = document.getElementById('dish-difficulty').value;
    const ingredients = document.getElementById('dish-ingredients').value;
    const steps = document.getElementById('dish-steps').value;
    
    // 显示加载状态
    const submitButton = document.querySelector('.submit-button');
    const originalText = submitButton.textContent;
    submitButton.textContent = '提交中...';
    submitButton.disabled = true;
    
    try {
        // 调用API提交菜品
        const response = await fetch(`${API_BASE_URL}/submit-dish`, {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json'
            },
            body: JSON.stringify({
                dish_name: dishName,
                category: category,
                difficulty: difficulty,
                ingredients: ingredients,
                steps: steps
            })
        });
        
        if (!response.ok) {
            throw new Error('提交失败');
        }
        
        const data = await response.json();
        
        // 恢复按钮状态
        submitButton.textContent = originalText;
        submitButton.disabled = false;
        
        // 显示成功消息
        const successMessage = document.createElement('div');
        successMessage.className = 'success-message';
        successMessage.textContent = '菜品提交成功！';
        submitForm.appendChild(successMessage);
        
        // 清空表单
        submitForm.reset();
        
        // 3秒后移除成功消息
        setTimeout(() => {
            successMessage.remove();
        }, 3000);
    } catch (error) {
        console.error('提交菜品失败:', error);
        
        // 恢复按钮状态
        submitButton.textContent = originalText;
        submitButton.disabled = false;
        
        // 显示错误消息
        const errorMessage = document.createElement('div');
        errorMessage.className = 'error-message';
        errorMessage.textContent = '提交失败，请稍后再试。';
        submitForm.appendChild(errorMessage);
        
        // 3秒后移除错误消息
        setTimeout(() => {
            errorMessage.remove();
        }, 3000);
    }
}

// 处理删除菜品
async function handleDeleteDish(e) {
    e.preventDefault();
    
    // 获取表单数据
    const dishName = document.getElementById('delete-dish-name').value;
    const category = document.getElementById('delete-dish-category').value;
    
    // 显示确认对话框
    if (!confirm(`确定要删除菜品 "${dishName}" 吗？`)) {
        return;
    }
    
    // 显示加载状态
    const deleteButton = document.querySelector('.delete-button');
    const originalText = deleteButton.textContent;
    deleteButton.textContent = '删除中...';
    deleteButton.disabled = true;
    
    try {
        // 调用API删除菜品
        const response = await fetch(`${API_BASE_URL}/delete-dish`, {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json'
            },
            body: JSON.stringify({
                dish_name: dishName,
                category: category
            })
        });
        
        if (!response.ok) {
            throw new Error('删除失败');
        }
        
        const data = await response.json();
        
        // 恢复按钮状态
        deleteButton.textContent = originalText;
        deleteButton.disabled = false;
        
        // 显示成功消息
        const successMessage = document.createElement('div');
        successMessage.className = 'success-message';
        successMessage.textContent = '菜品删除成功！';
        deleteForm.appendChild(successMessage);
        
        // 清空表单
        deleteForm.reset();
        
        // 3秒后移除成功消息
        setTimeout(() => {
            successMessage.remove();
        }, 3000);
    } catch (error) {
        console.error('删除菜品失败:', error);
        
        // 恢复按钮状态
        deleteButton.textContent = originalText;
        deleteButton.disabled = false;
        
        // 显示错误消息
        const errorMessage = document.createElement('div');
        errorMessage.className = 'error-message';
        errorMessage.textContent = '删除失败，请稍后再试。';
        deleteForm.appendChild(errorMessage);
        
        // 3秒后移除错误消息
        setTimeout(() => {
            errorMessage.remove();
        }, 3000);
    }
}

// 初始化应用
init();