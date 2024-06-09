FROM python:3.9-slim

WORKDIR /app

COPY requirements.txt requirements.txt
COPY api.py api.py
COPY modelo.py modelo.py

RUN pip install --no-cache-dir -r requirements.txt

EXPOSE 8080

CMD ["python", "api.py"]
