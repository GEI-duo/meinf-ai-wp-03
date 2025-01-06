FROM python:3.9.13-slim

WORKDIR /app/fastapi

COPY fastapi/requirements.txt /app/fastapi/
RUN pip install --no-cache-dir --upgrade -r /app/fastapi/requirements.txt

COPY ../cache/models/pipeline.pkl /app/cache/models/pipeline.pkl
COPY fastapi/main.py /app/fastapi/

EXPOSE 8000

CMD ["fastapi", "run", "main.py", "--port", "8000"]

