# 使用一个兼容性较好的 Python 官方镜像
FROM python:3.11-slim

# 设置工作目录
WORKDIR /app

# 1. 先复制本地预下载的所有包
COPY packages/ /tmp/packages/

# 2. 复制依赖锁定文件
COPY requirements_lock.txt .

# 3. 从本地离线安装依赖（极大加速！）
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir --find-links=/tmp/packages -r requirements_lock.txt && \
    # 如果本地包不完整,仍可从网络补充
    pip install --no-cache-dir -r requirements_lock.txt || true

# 4. 清理临时文件以减小镜像体积（可选）
RUN rm -rf /tmp/packages

# 5. 复制应用程序源代码
COPY . .

# 6. 声明 Gradio 默认端口
EXPOSE 7860

# 7. 设置容器启动命令
CMD ["python", "app.py"]