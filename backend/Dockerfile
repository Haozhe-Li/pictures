FROM python:3.10.9

ENV PYTHONUNBUFFERED True
ENV APP_HOME /app
WORKDIR $APP_HOME
COPY . ./

# RUN apt-get update && apt-get install -y libgl1 libglib2.0-0 ffmpeg libsm6 libxext6

ENV PORT 8080

RUN pip install --no-cache-dir -r requirements.txt

# TrustMark's ONNX models (core/watermark), fetched once here so watermarking never waits on a
# download; they are loaded into memory only while a photo is watermarked or checked.
RUN python -m core.watermark

CMD exec uvicorn main:app --host 0.0.0.0 --port ${PORT}
