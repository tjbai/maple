from abc import ABC, abstractmethod
from typing import List
from uuid import uuid4

import PyPDF2 as pdf
import tiktoken
from langchain.text_splitter import RecursiveCharacterTextSplitter
from models import ChunkMetadata, ChunkWrapper, PineconeConfig, UserFilePage
from tqdm import tqdm


class LoaderContext:
    def __init__(self, pc_config: PineconeConfig, file, file_from: str):
        switch = {
            "pdf": PDFLoader,
        }

        loader = switch.get(file_from, None)
        if loader is None:
            raise Exception("unsupported file type")

        self.loader = loader(file)
        self.pc_config = pc_config

    def load(self):
        # load, add to db, and return
        pass

    def upsert(self):
        # calculate embeddings and add to index:namesapce
        pass


class Loader(ABC):
    def __init__(self, file):
        self.file = file
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=500,
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
        with open(self.file, "rb") as f:
            pdfReader = pdf.PdfReader(f)

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
                    ids=[uuid4() for _ in range(len(text_chunks))],
                    page_no=i + 1,
                )

                pages.append(new_page)
                base_index += len(text_chunks)

            return pages
