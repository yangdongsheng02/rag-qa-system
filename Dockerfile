# 使用一个兼容性较好的 Python 官方镜像
FROM python:3.11-slim

# 设置工作目录
WORKDIR /app

# 1. 复制依赖锁定文件
COPY requirements_lock.txt .

# 2. 使用国内镜像源安装依赖（此处以阿里云为例，可替换）
RUN pip install --no-cache-dir -r requirements_lock.txt \
    -i https://mirrors.aliyun.com/pypi/simple/ \
    --trusted-host mirrors.aliyun.com

# 3. 复制应用程序源代码
COPY . .

# 4. 声明 Gradio 默认端口
EXPOSE 7860

# 5. 设置容器启动命令
CMD ["python", "app.py"]