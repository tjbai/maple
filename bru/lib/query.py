from typing import List, Tuple

import pinecone
from langchain.chains import RetrievalQA
from langchain.chains.summarize import load_summarize_chain
from langchain.chat_models import ChatOpenAI
from langchain.docstore.document import Document
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.prompts import PromptTemplate
from langchain.vectorstores import Pinecone

from .models import OpenAIConfig, PineconeConfig
from .system import map_template, reduce_template


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
            namespace=pc_config.namespace,
        )

        """
        # todo: look into different citation methods
        # todo: look into conversational retrieval 
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

        self.summ = load_summarize_chain(
            ChatOpenAI(
                openai_api_key=oai_config.api_key,
                model_name=oai_config.summary_model,
                temperature=0.0,
            ),
            chain_type="map_reduce",
            return_intermediate_steps=True,
            map_prompt=PromptTemplate(input_variables=["text"], template=map_template),
            combine_prompt=PromptTemplate(
                input_variables=["text"], template=reduce_template
            ),
        )

    def query(self, q: str) -> Tuple[str, str]:
        result = self.qa({"query": q})
        return (result["result"], result["source_documents"])

    def summarize(self, chunks: List[str]) -> Tuple[str, str, str]:
        res = self.summ([Document(page_content=chunk) for chunk in chunks])
        return (res["input_documents"], res["intermediate_steps"], res["output_text"])
