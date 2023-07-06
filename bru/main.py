import json
import os
from typing import Annotated

from fastapi import FastAPI, File, Form, UploadFile
from lib.load import LoaderContext
from lib.models import OpenAIConfig, PineconeConfig
from lib.query import QueryBuilder
from pydantic import BaseModel

app = FastAPI()


@app.get("/api/hello/")
def hello():
    return {"dick": "cock"}


@app.post("/api/load-file/")
async def load_file(
    file: Annotated[UploadFile, File()], namespace: Annotated[str, Form()]
):  # extract, upload to metadata/vector db, return
    oai_config = OpenAIConfig(api_key=os.environ.get("OPENAI_API_KEY"))
    pc_config = PineconeConfig(
        "prob",
        os.environ.get("PINECONE_API_KEY"),
        "asia-southeast1-gcp-free",
        namespace=namespace,
    )
    loader = LoaderContext(pc_config, oai_config, file.file, file.content_type)

    pages = loader.execute()
    return {"pages": pages}


@app.post("/api/summarize/")
def summarize():
    return


@app.post("/api/query/")
def query():
    return
