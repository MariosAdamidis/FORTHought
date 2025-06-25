# Dockerfile for Unsloth with JupyterLab on ROCm/WSL (Enhanced Academic Stack)
# v4.0: Added PDF/OCR, cheminformatics, scientific libraries

FROM rocm/pytorch:rocm6.4_ubuntu22.04_py3.10_pytorch_release_2.6.0
WORKDIR /root

#--- System Dependencies -----------------------------------------------------
# Build tools, math libs, PDF utilities, OCR, Java for Tabula
RUN apt-get update && apt-get install -y \
    git cmake libpq-dev \
    build-essential libopenblas-dev liblapack-dev \
    ghostscript poppler-utils openjdk-17-jre-headless \
    tesseract-ocr tesseract-ocr-eng \
 && rm -rf /var/lib/apt/lists/*

#--- Unsloth & ROCm Dependencies ------------------------------------------------
RUN git clone -b rocm_enabled_multi_backend https://github.com/ROCm/bitsandbytes.git && \
    cd bitsandbytes && \
    cmake -DGPU_TARGETS="gfx1100" -DBNB_ROCM_ARCH="gfx1100" -DCOMPUTE_BACKEND=hip -S . && \
    make -j$(nproc) && pip install . && cd .. && rm -rf bitsandbytes

RUN pip install --no-cache-dir \
    unsloth_zoo>=2025.5.7 datasets>=3.4.1 sentencepiece>=0.2.0 \
    tqdm psutil wheel>=0.42.0 accelerate>=0.34.1 peft>=0.7.1,!=0.11.0

RUN git clone https://github.com/ROCm/xformers.git && \
    cd xformers && git submodule update --init --recursive && git checkout 13c93f3 && \
    PYTORCH_ROCM_ARCH=gfx1100 python setup.py install && cd .. && rm -rf xformers

ENV FLASH_ATTENTION_TRITON_AMD_ENABLE="TRUE"
RUN git clone https://github.com/ROCm/flash-attention.git && \
    cd flash-attention && git checkout main_perf && python setup.py install && cd .. && rm -rf flash-attention

ENV LD_LIBRARY_PATH=/usr/lib/wsl/lib:$LD_LIBRARY_PATH

RUN git clone https://github.com/unslothai/unsloth.git && \
    cd unsloth && pip install . && cd .. && rm -rf unsloth

#--- Core Python Scientific & Office Stack -------------------------------------
RUN pip install --no-cache-dir \
    jupyterlab notebook pandas numpy matplotlib seaborn plotly \
    scipy scikit-learn statsmodels scikit-image dask[complete] xarray netCDF4 \
    requests openpyxl xlrd Pillow python-docx python-pptx \
    rdkit ase \
    # PDF/OCR tools
    pymupdf pdfplumber camelot-py[cv] tabula-py pdf2image pytesseract pdfminer.six \
    # Profiling & optimization
    memory_profiler line_profiler numba

#--- Jupyter Configuration --------------------------------------------------
RUN mkdir -p /root/.jupyter && \
    echo "c.ServerApp.token = ''" > /root/.jupyter/jupyter_server_config.py && \
    echo "c.ServerApp.password = ''" >> /root/.jupyter/jupyter_server_config.py && \
    echo "c.ServerApp.disable_check_xsrf = True" >> /root/.jupyter/jupyter_server_config.py

EXPOSE 8888
env GRANT_SUDO="yes"
ENV NB_USER="root"

CMD ["jupyter", "lab", "--ip=0.0.0.0", "--port=8888", "--no-browser", "--allow-root", "--notebook-dir=/data"]
