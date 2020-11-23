FROM python:3.8
ARG RUN_ID
ARG MODEL_NAME
RUN pip install mlflow
RUN pip install tensorflow
RUN pip install flask
RUN pip install pandas
RUN pip install boto3
RUN pip install scipy
RUN pip install librosa
COPY requirements.txt .
RUN pip install -r requirements.txt
COPY common.py .
COPY config.yaml .
ENV SERVER_HOST 0.0.0.0
ENV SERVER_PORT 5005
ENV MINIO_ACCESS_KEY minio
ENV MINIO_SECRET_KEY miniostorage
ENV MLFLOW_S3_ENDPOINT_URL http://172.17.0.2:9000
ENV AWS_ACCESS_KEY_ID minio
ENV AWS_SECRET_ACCESS_KEY miniostorage
ENV ARTIFACT_STORE s3://test/0/7b3cc8e4d457455e806ed877f099efe1/artifacts/model
COPY mlfow_load_model_and_run.py .
CMD ["python","-u","mlfow_load_model_and_run.py"]

