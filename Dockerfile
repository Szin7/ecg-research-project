FROM python:3.12-slim
ENV PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1 OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 PYTHONPATH=/opt/ecg-research-project
WORKDIR /opt/ecg-research-project
COPY requirements-cpu.lock ./
RUN python -m pip install --no-cache-dir --require-hashes -r requirements-cpu.lock
COPY pyproject.toml README.md ./
COPY src ./src
COPY app ./app
COPY scripts ./scripts
COPY configs ./configs
COPY artifacts ./artifacts
COPY data/demo ./data/demo
COPY data/preview ./data/preview
COPY .streamlit ./.streamlit
RUN python -m pip install --no-deps --no-build-isolation . \
    && python -m scripts.prepare_data --mode demo --task both \
    && useradd --create-home --uid 10001 ecg-research-project \
    && chown -R ecg-research-project:ecg-research-project /opt/ecg-research-project
USER ecg-research-project
EXPOSE 8501
HEALTHCHECK --interval=30s --timeout=5s --start-period=45s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://127.0.0.1:8501/_stcore/health',timeout=3)"
CMD ["python","-m","streamlit","run","app/main.py","--server.address=0.0.0.0","--server.port=8501","--server.headless=true"]
