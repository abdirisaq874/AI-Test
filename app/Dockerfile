FROM openfabric/tee-python-cpu:dev

# Copy Poetry config files only
COPY pyproject.toml poetry.lock* ./

# Upgrade pip and install Poetry
RUN python3 -m pip install --upgrade pip && \
    python3 -m pip install --upgrade poetry && \
    poetry config virtualenvs.create false && \
    poetry install --only main && \
    rm -rf ~/.cache/pypoetry/{cache,artifacts} ~/.cache/pip

# Install additional Python packages
RUN python3 -m pip install \
    torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118 && \
    transformers \
    streamlit \
    langchain \
    langchain_community \
    sentence-transformers \
    accelerate \
    faiss-gpu \
    faiss-cpu \
    chromadb

# Copy the full application source code
COPY . .

# Expose the port for Streamlit
EXPOSE 8501

# Environment variables for Streamlit
ENV STREAMLIT_SERVER_PORT=8501
ENV STREAMLIT_SERVER_HEADLESS=true
ENV STREAMLIT_SERVER_ADDRESS=0.0.0.0

# Start Streamlit application
CMD ["streamlit", "run", "main.py"]
