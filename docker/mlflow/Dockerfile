FROM python:3.8-slim-buster

RUN pip install mlflow[extras]

EXPOSE 5000

ENTRYPOINT ["mlflow", "server"]

CMD ["--host", "0.0.0.0", "--backend-store-uri", "sqlite:///mlflow.db", "--default-artifact-root", "/mlflow"]
