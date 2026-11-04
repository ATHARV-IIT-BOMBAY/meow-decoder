FROM python:3.12-slim
RUN apt-get update && apt-get install -y --no-install-recommends ffmpeg git && rm -rf /var/lib/apt/lists/*
RUN useradd -m -u 1000 user
USER user
ENV HOME=/home/user PATH=/home/user/.local/bin:$PATH MEOW_HOME=/tmp/meow PYTHONUNBUFFERED=1
WORKDIR /home/user/app
RUN pip install --no-cache-dir --user torch --index-url https://download.pytorch.org/whl/cpu
RUN pip install --no-cache-dir --user "meow-decoder @ git+https://github.com/ATHARV-IIT-BOMBAY/meow-decoder@main"
# Download weights and both audio models at build time so the Space starts fast
RUN python -c "import numpy as np; from meow_decoder import Decoder; Decoder('beats').embed(np.zeros(16000, np.float32))"
EXPOSE 7860
CMD ["meow", "serve", "--host", "0.0.0.0", "--port", "7860", "--no-contrib"]
