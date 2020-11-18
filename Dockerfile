FROM  python:3.8.5
ARG RUN_ID
ARG MODEL_NAME
#RUN pip install mlflow
#RUN pip install tensorflow
#RUN pip install boto3
COPY requirements.txt .
RUN pip install -r requirements.txt
ENV SERVER_PORT 5003
ENV SERVER_HOST 0.0.0.0
ENV MINIO_ACCESS_KEY=minio
ENV MINIO_SECRET_KEY=miniostorage
ENV MLFLOW_S3_ENDPOINT_URL=http://172.17.0.2:9000
ENV AWS_ACCESS_KEY_ID=minio
ENV AWS_SECRET_ACCESS_KEY=miniostorage
ENV ARTIFACT_STORE s3://test/0/03c1468909144a8e918fe88a75f27bb1/artifacts/model
COPY run_mlflow_model_serve.sh .
#RUN chmod +x run_mlflow_model_serve.sh
ENTRYPOINT  ["bash", "run_mlflow_model_serve.sh"]