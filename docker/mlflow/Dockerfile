FROM python:3.8-slim-buster


EXPOSE 5000

ENTRYPOINT ["mlflow", "server", \
            "--host", "0.0.0.0", \
            "--backend-store-uri", "sqlite:///mlflow.db", \
            "--default-artifact-root", "/mlflow"]
