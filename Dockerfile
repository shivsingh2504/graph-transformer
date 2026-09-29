FROM python:3.11-slim

WORKDIR /app
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

COPY requirements_clean.txt ./requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

COPY src ./src
COPY ui/app.py ./ui/app.py
COPY .streamlit/config.toml ./.streamlit/config.toml
COPY checkpoints_run8/results.txt ./checkpoints_run8/results.txt
COPY scripts/ensure_checkpoint.py ./scripts/ensure_checkpoint.py
COPY scripts/start_api.sh ./scripts/start_api.sh
COPY scripts/start_render_free.sh ./scripts/start_render_free.sh
RUN chmod +x ./scripts/start_api.sh ./scripts/start_render_free.sh

EXPOSE 8000 8501
CMD ["/app/scripts/start_render_free.sh"]
