FROM python:3.9-slim

WORKDIR /app

RUN pip install --no-cache-dir \
    numpy \
    pandas \
    scipy \
    scikit-learn \
    statsmodels 

COPY . /app

CMD ["python", "trade.py"]