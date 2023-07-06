from dataclasses import dataclass
from typing import Tuple

import pinecone
from langchain.chains import RetrievalQA
from langchain.chat_models import ChatOpenAI
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.vectorstores import Pinecone


@dataclass
class PineconeConfig:
    index_name: str
    api_key: str
    env: str


@dataclass
class OpenAIConfig:
    api_key: str
    embedding_model: str = "text-embedding-ada-002"
    completion_model: str = "gpt-3.5-turbo"
    condense_model: str = "gpt-3.5-turbo"


class QueryBuilder:
    def __init__(self, pc_config: PineconeConfig, oai_config: OpenAIConfig):
        self.pc_config = pc_config
        self.oai_config = oai_config

        pinecone.init(api_key=pc_config.api_key, environment=pc_config.env)

        vectorstore = Pinecone.from_existing_index(
            pc_config.index_name,
            OpenAIEmbeddings(
                model=oai_config.embedding_model, openai_api_key=oai_config.api_key
            ),
        )

        """
        # todo: look into different citation methods
        """

        """
        # todo: can't get chat history to work...
        self.qa = ConversationalRetrievalChain.from_llm(
            ChatOpenAI(
                openai_api_key=oai_config.api_key,
                model_name=oai_config.completion_model,
                temperature=0.0,
            ),
            vectorstore.as_retriever(),
            condense_question_llm=ChatOpenAI(
                openai_api_key=oai_config.api_key,
                model_name=oai_config.condense_model,
                temperature=0.0,
            ),
            return_source_documents=True,
            get_chat_history=lambda h: h,
        )
        """

        self.qa = RetrievalQA.from_chain_type(
            llm=ChatOpenAI(
                openai_api_key=oai_config.api_key,
                model_name=oai_config.completion_model,
                temperature=0.0,
            ),
            chain_type="stuff",
            retriever=vectorstore.as_retriever(),
            return_source_documents=True,
        )

    def query(self, q: str) -> Tuple[str, str]:
        """
        # todo: figure out chat history

        result: Dict[Literal['question', 'answer', 'chat_history', 'source_documents'], Any]
            = self.qa({'question': q, 'chat_history', chat_history})
        """

        result = self.qa({"query": q})
        return (result["result"], result["source_documents"])
