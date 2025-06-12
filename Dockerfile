FROM python:3.11-slim

WORKDIR /app

COPY requirements.txt ./
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

EXPOSE 8000

ENV LLM_MODE="api"

CMD ["sh", "-c", "python3.11 main.py --mode api"]