FROM python:3.9.13-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache --upgrade -r requirements.txt

COPY main.py .

EXPOSE 7860
ENV GRADIO_SERVER_NAME="0.0.0.0"

CMD ["python", "main.py"]
