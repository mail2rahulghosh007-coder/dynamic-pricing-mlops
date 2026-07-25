FROM python:3.10-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

EXPOSE 7860

CMD ["sh", "-c", "uvicorn src.api:app --host 0.0.0.0 --port 8000 & streamlit run src/streamlit_app.py --server.port 7860 --server.address 0.0.0.0"]