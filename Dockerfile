FROM --platform=linux/amd64 python:3.10-slim-bullseye

WORKDIR /app

RUN uname -m

ENV PIP_DEFAULT_TIMEOUT=1000
ENV PIP_RETRIES=10
ENV PYTHONUNBUFFERED=1

RUN apt-get update && apt-get install -y \
    build-essential \
    curl \
    git \
    libgomp1 \
    python3-dev \
    && rm -rf /var/lib/apt/lists/*

RUN pip install --upgrade "setuptools<70.0.0" pip wheel Cython

# ============================================================================
# MAIN ENVIRONMENT
# ============================================================================

RUN pip install --no-cache-dir --prefer-binary \
    numpy==1.26.4 \
    pandas==2.1.4 \
    scikit-learn==1.3.2 \
    scipy==1.14.1 \
    matplotlib==3.8.3 \
    seaborn==0.13.2 \
    openpyxl==3.1.2

RUN pip install --no-cache-dir --prefer-binary \
    prophet==1.1.5 \
    catboost==1.2.3

RUN pip install --no-cache-dir --prefer-binary \
    xgboost==2.0.3 \
    lightgbm==4.3.0

RUN pip install --no-cache-dir --prefer-binary \
    tsfresh==0.21.0 \
    tslearn==0.6.3 \
    dtaidistance==2.3.13 \
    pingouin==0.5.5 \
    dieboldmariano==1.1.0

RUN pip install --no-cache-dir --prefer-binary \
    "https://github.com/etna-team/etna/archive/refs/tags/2.10.0.tar.gz#egg=etna"

RUN pip install --no-cache-dir "setuptools<70.0.0"

COPY . /app/

CMD ["python", "generate_presentation_plots.py"]