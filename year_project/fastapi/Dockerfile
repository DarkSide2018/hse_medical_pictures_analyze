FROM python:3.10-slim

WORKDIR /code

RUN apt-get update \
    && apt-get install -y libgl1-mesa-glx \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

RUN apt-get update && apt-get install -y libglib2.0-0 libsm6 libxrender1 libxext6


RUN pip3 install fastapi uvicorn redis pandas numpy pillow opencv-python scikit-image scikit-learn python-multipart

COPY ./app /code

CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
