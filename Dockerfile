FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PYTHONPATH=/app/src \
    STREAMLIT_SERVER_HEADLESS=true \
    STREAMLIT_BROWSER_GATHER_USAGE_STATS=false

WORKDIR /app

# Copy and install dependencies
COPY requirements.txt .

# Install torch CPU-only first to avoid 2GB CUDA download
RUN pip install --upgrade pip && \
    pip install -r requirements.txt

# Install the package itself so "python -m aero_fusion.X" works
COPY src ./src
COPY setup.py ./
RUN pip install -e . --no-deps 2>/dev/null || true

# Copy application files
COPY demo_app.py ./
COPY README.md ./
COPY scripts ./scripts
COPY docs ./docs
COPY notebooks ./notebooks

# Copy only the artifact folders needed to run the demo
# (model weights, cleaned catalog, test results)
# These are copied from your local disk - no internet download needed
RUN mkdir -p artifacts/step2_clean/flights
COPY artifacts/step2_clean/catalog ./artifacts/step2_clean/catalog
COPY artifacts/step2_clean/flights/20250801_4007f1_173814_202441 ./artifacts/step2_clean/flights/20250801_4007f1_173814_202441
COPY artifacts/step2_clean/flights/20231114_485341_032118_043049 ./artifacts/step2_clean/flights/20231114_485341_032118_043049
COPY artifacts/step2_clean/flights/20240827_4b187f_064414_074830 ./artifacts/step2_clean/flights/20240827_4b187f_064414_074830
COPY artifacts/step2_clean/flights/20231011_a0f427_041753_051756 ./artifacts/step2_clean/flights/20231011_a0f427_041753_051756
COPY artifacts/step2_clean/flights/20250703_4005c0_202353_224235 ./artifacts/step2_clean/flights/20250703_4005c0_202353_224235
COPY artifacts/step2_clean/flights/20250326_4006c1_151222_174303 ./artifacts/step2_clean/flights/20250326_4006c1_151222_174303
COPY artifacts/step2_clean/flights/20240822_ac21af_054936_065856 ./artifacts/step2_clean/flights/20240822_ac21af_054936_065856
COPY artifacts/step2_clean/flights/20231105_a0a54d_030849_041552 ./artifacts/step2_clean/flights/20231105_a0a54d_030849_041552
COPY artifacts/step2_clean/flights/20240710_4007f0_092932_103853 ./artifacts/step2_clean/flights/20240710_4007f0_092932_103853
COPY artifacts/step2_clean/flights/20250318_400773_201647_221532 ./artifacts/step2_clean/flights/20250318_400773_201647_221532
COPY artifacts/step4_ml_dataset/catalog ./artifacts/step4_ml_dataset/catalog
COPY artifacts/step5_gru ./artifacts/step5_gru
COPY artifacts/step5_kalman ./artifacts/step5_kalman
COPY artifacts/step3_baseline/catalog ./artifacts/step3_baseline/catalog

EXPOSE 8501

CMD ["streamlit", "run", "demo_app.py", "--server.address=0.0.0.0", "--server.port=8501"]
