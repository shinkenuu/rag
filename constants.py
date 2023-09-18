from os import environ

WEAVIATE_URL = environ.get("WEAVIATE_URL", "http://127.0.0.1:8080")
