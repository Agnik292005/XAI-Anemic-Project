FROM python:3.13-slim

WORKDIR /app

ENV PYTHONUNBUFFERED=1
ENV PIP_NO_CACHE_DIR=1
ENV MPLCONFIGDIR=/tmp/matplotlib

RUN pip install --no-cache-dir \
    --index-url https://download.pytorch.org/whl/cpu \
    torch torchvision

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY modeling_podnet.py app.py final_summary_run2.json ./
COPY checkpoints ./checkpoints
COPY assets ./assets

ENV PORT=8080
EXPOSE 8080

CMD ["python", "app.py"]
