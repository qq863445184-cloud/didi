FROM python:3.11-slim

WORKDIR /app

COPY weather-mcp-server/requirements.txt ./requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

COPY weather-mcp-server/server.py ./server.py
COPY app ./app

CMD ["python", "server.py"]
