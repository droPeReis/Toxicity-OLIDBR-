FROM pytorch/pytorch:1.13.1-cuda11.6-cudnn8-runtime

WORKDIR /app

COPY /src /app/src
COPY /requirements.txt /app

RUN pip install --upgrade pip setuptools wheel && \
    pip install -r requirements.txt && \
    pip install -r src/ml/requirements.txt

ENV PORT 8080

EXPOSE 8080

# alembic upgrade head &&
CMD ["sh", "-c", "uvicorn src.api.main:app --host 0.0.0.0 --port $PORT"]
