FROM python:3.10

WORKDIR /app

RUN apt-get update && apt-get install -y postgresql-client

COPY requirements.txt .

RUN pip install --no-cache-dir -r requirements.txt

COPY . .

CMD ["uvicorn", "interface.main:app", "--host", "0.0.0.0", "--port", "8000"]

EXPOSE 8501
