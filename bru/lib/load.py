from abc import ABC, abstractmethod
from typing import List
from uuid import uuid4

import numpy as np
import pinecone
import PyPDF2 as pdf
import tiktoken
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from tqdm import tqdm

from .models import ChunkMetadata, OpenAIConfig, PineconeConfig, UserFilePage


class LoaderContext:
    def __init__(
        self,
        pc_config: PineconeConfig,
        oai_config: OpenAIConfig,
        file,
        content_type: str,
    ):
        switch = {
            "application/pdf": PDFLoader,
            "application/txt": TXTLoader,
        }

        loader = switch.get(content_type, None)
        if loader is None:
            raise Exception("unsupported file type")

        self.loader = loader(file)
        self.pc_config = pc_config
        self.oai_config = oai_config

        pinecone.init(api_key=pc_config.api_key, environment=pc_config.env)
        self.index = pinecone.Index(
            pc_config.index_name
        )  # GRPCIndex for multi-pod setups

    def execute(self):
        pages = self.loader.load()
        self.__upsert(pages)
        return "success!"

    def __upsert(self, pages: list[UserFilePage]):
        metadatas, ids, texts = [], [], []
        for page in pages:
            ids.extend([id for id in page.ids])
            texts.extend([m.text for m in page.metadatas])
            metadatas.extend(
                [
                    {"index": m.index, "text": m.text, "token_length": m.token_length}
                    for m in page.metadatas
                ]
            )

        embeddings = [list(e) for e in self.__compute_embeddings(texts)]
        assert len(embeddings) == len(ids) == len(metadatas)

        vectors = [
            (id, embedding, metadata)
            for id, embedding, metadata in zip(ids, embeddings, metadatas)
        ]

        for i in tqdm(range(0, len(vectors), 100)):
            slice = vectors[i : i + 100]
            self.index.upsert(vectors=slice, namespace=self.pc_config.namespace)

    def __compute_embeddings(self, chunks: List[str]) -> List[np.ndarray]:
        encoder = OpenAIEmbeddings(
            model=self.oai_config.embedding_model,
            openai_api_key=self.oai_config.api_key,
        )
        embeddings = encoder.embed_documents(chunks)
        return [np.array(e) for e in embeddings]


class Loader(ABC):
    def __init__(self, file):
        self.file = file
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=400,
            chunk_overlap=20,
            length_function=self.token_length,
            separators=["\n\n", "\n", " ", ""],
        )

    def token_length(self, chunk: str) -> int:
        tokenizer = tiktoken.get_encoding("cl100k_base")
        tokens = tokenizer.encode(chunk, disallowed_special=())
        return len(tokens)

    @abstractmethod
    def load(self) -> List[UserFilePage]:
        return


class PDFLoader(Loader):
    def load(self) -> List[UserFilePage]:
        # with open(self.file, "rb") as f:
        pdfReader = pdf.PdfReader(self.file)

        pages = []
        base_index = 0
        for i, page in tqdm(enumerate(pdfReader.pages)):
            text = page.extract_text()
            text_chunks = self.text_splitter.split_text(text)

            new_page = UserFilePage(
                metadatas=[
                    ChunkMetadata(
                        index=base_index + i,
                        text=text,
                        token_length=self.token_length(text),
                    )
                    for i, text in enumerate(text_chunks)
                ],
                ids=[str(uuid4()) for _ in range(len(text_chunks))],
                page_no=i,
            )

            pages.append(new_page)
            base_index += len(text_chunks)

        return pages


class TXTLoader(Loader):
    pass
