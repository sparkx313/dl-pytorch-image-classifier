FROM python:3.11-slim
WORKDIR /app
COPY requirements.txt .
RUN python -m pip install --no-cache-dir -r requirements.txt
COPY classifier.py .
ENTRYPOINT ["python", "classifier.py"]
