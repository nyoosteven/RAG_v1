import os
import sys
import json
import copy
import openai
import pickle
import nltk
import nest_asyncio
from tqdm import tqdm
from llama_index.core import (VectorStoreIndex,
                         ServiceContext,
                         set_global_service_context,
                         SimpleKeywordTableIndex)

from llama_index.core.retrievers import RecursiveRetriever
from llama_index.core.query_engine import RetrieverQueryEngine
from llama_index.llms.openai import OpenAI
from llama_index.core.callbacks import CallbackManager, LlamaDebugHandler
from pathlib import Path
from llama_index.core.tools import ToolMetadata, QueryEngineTool
from llama_index.core.objects import ObjectIndex, SimpleToolNodeMapping
from llama_index.agent.openai_legacy import OpenAIAgent, FnRetrieverOpenAIAgent
from llama_index.core import SimpleKeywordTableIndex
from build_vector_db import build_vector_db

nest_asyncio.apply()
llama_debug = LlamaDebugHandler(print_trace_on_end=True)
callback_manager = CallbackManager([llama_debug])
os.environ["OPENAI_API_KEY"] = ""

nltk.data.path.append('nltk_data')

llm = OpenAI(temperature=0, 
             model="gpt-3.5-turbo",
             callback_manager= callback_manager,
             max_token = 1000)

num_output = 256
chunk_size_limit = 1000
service_context = ServiceContext.from_defaults(num_output=num_output, 
                                               chunk_size_limit=chunk_size_limit, 
                                               llm=llm)

set_global_service_context(service_context)

prompt="Useful for retrieving about document {file_name}"

class MultiDocumentQueryEngine():

    def __init__(self,):
        self.agents_dict = {}
        self.query_engine = {}
        self.prod_qe = []
        self.pymupdf = build_vector_db()
        self.all_nodes = []

    def build_document_qe(self, pdf_folder, nodes_folder, summary_folder, file):
        
        file_name = str(file.split('.pdf')[0]).lower()
        
        pdf_path = f'{pdf_folder}/{file_name}.pdf'
        nodes_path = f'{nodes_folder}/{file_name}.pkl'
        summary_path = f'{summary_folder}/{file_name}.txt'

        base_nodes, summary = self.pymupdf.get_nodes_from_documents(pdf_path, file_name, nodes_path, summary_path)
        self.all_nodes.extend(base_nodes)

        retriever = VectorStoreIndex(base_nodes).as_retriever(similarity_top_k=5)
        query_engine = RetrieverQueryEngine.from_args(retriever)

        qe_tool = QueryEngineTool(
                        query_engine = query_engine,
                        metadata = ToolMetadata(
                            name = f"tool_{file_name.replace(' ','_')}",
                            description = prompt.format(file_name=file_name)+summary,
                        )
                    )
        
        return qe_tool
    
    def build_query_engine(self, html_folder, nodes_folder, summary_folder, model_path):

        if model_path is None or not os.path.exists(model_path):
            for file in tqdm(os.listdir(html_folder)):
                
                qe_tool = self.build_document_qe(pdf_folder=html_folder,
                                    nodes_folder=nodes_folder,
                                    summary_folder=summary_folder,
                                    file=file)
                
                self.prod_qe.append(qe_tool)
            
            keyword_index = SimpleKeywordTableIndex(self.all_nodes)
            self.keyword_query_engine = keyword_index.as_query_engine(service_context=service_context)
            self.prod_qe.append(QueryEngineTool.from_defaults(query_engine=self.keyword_query_engine,
                                                        description="Useful for retrieving specific context using keywords",))
            
            pickle.dump(self.prod_qe, open(model_path,'wb'))
        else:
            self.prod_qe = pickle.load(open(model_path,"rb"))
        
        tool_mapping = SimpleToolNodeMapping.from_objects(self.prod_qe)
        obj_index = ObjectIndex.from_objects(self.prod_qe,
                                                tool_mapping,
                                                VectorStoreIndex)
        system_prompt=""" \
        You are an agent designed to answer queries about a set of given documents.
        Please always use the tools provided to answer a question. Do not rely on prior knowledge.\
        """
        top_agent = FnRetrieverOpenAIAgent.from_retriever(obj_index.as_retriever(similarity_top_k=5),
                                                        llm=llm,
                                                        system_prompt=system_prompt,
                                                        verbose=True)
        return top_agent

    def build_query_engine_document(self, raw_nodes, node_mappings):
        """
        Build retriever Query Engine
        """
        vector_index = VectorStoreIndex(raw_nodes)
        vector_retriever = vector_index.as_retriever(similarity_top_k=3)
        recursive_retriever = RecursiveRetriever(
            "vector",
            retriever_dict={"vector": vector_retriever},
            node_dict=node_mappings,
            verbose=True,
        )
        query_engine = RetrieverQueryEngine.from_args(recursive_retriever)
        return query_engine


