FROM python:3.11-slim

WORKDIR /app

# Install dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy source
COPY src/ ./src/
COPY app.py .
COPY baseline.py .
COPY openenv.yaml .

# HF Spaces runs on port 7860
ENV GRADIO_SERVER_PORT=7860
ENV GRADIO_SERVER_NAME=0.0.0.0
ENV PYTHONPATH=/app/src

EXPOSE 7860

CMD ["python", "app.py"]
