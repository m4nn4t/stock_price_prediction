FROM python:3.10-slim

# set working directory
WORKDIR /app

# system deps (optional, but often helpful)
RUN apt-get update && apt-get install -y \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# copy files
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

# expose port
EXPOSE 8000

# run the app
CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8000"]
