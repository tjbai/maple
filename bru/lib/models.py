from dataclasses import dataclass
from typing import List, Tuple

import numpy as np

"""
Utility
"""


@dataclass
class PineconeConfig:
    index_name: str
    api_key: str
    env: str
    namespace: str = ""


@dataclass
class OpenAIConfig:
    api_key: str
    embedding_model: str = "text-embedding-ada-002"
    completion_model: str = "gpt-3.5-turbo"
    condense_model: str = "gpt-3.5-turbo"
    summary_model: str = "gpt-3.5-turbo"


"""
Data model
"""


@dataclass
class ChunkMetadata:
    index: int
    text: str
    token_length: int


@dataclass
class UserFilePage:
    metadatas: List[ChunkMetadata]
    ids: List[str]
    page_no: int
