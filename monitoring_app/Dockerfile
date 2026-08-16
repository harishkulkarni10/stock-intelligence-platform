FROM python:3.11-slim
WORKDIR /app
RUN pip install --no-cache-dir streamlit requests pandas
COPY app.py .
EXPOSE 8502
CMD ["streamlit", "run", "app.py", "--server.port=8502", "--server.address=0.0.0.0"]
