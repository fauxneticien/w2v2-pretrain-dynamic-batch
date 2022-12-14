FROM nvcr.io/nvidia/pytorch:22.11-py3

COPY requirements.txt /tmp/

RUN pip install --requirement /tmp/requirements.txt

