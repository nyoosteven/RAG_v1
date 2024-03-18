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
                         QueryBundle,
                         set_global_service_context,
                         Settings)

from llama_index.core.retrievers import RecursiveRetriever, RouterRetriever
from llama_index.llms.openai import OpenAI
from llama_index.core.callbacks import CallbackManager, LlamaDebugHandler
from pathlib import Path
from llama_index.core.tools import RetrieverTool
from llama_index.core.selectors import PydanticMultiSelector
from build_vector_db import PymuPDF
from llama_index.core import SimpleKeywordTableIndex
from llama_index.program.openai import OpenAIPydanticProgram
from llama_index.retrievers.bm25 import BM25Retriever
from llama_index.core import Settings

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

prompt_type = {
    'prospektus':"This is the prospectus document {file_name} that investors must carefully review, comprehend, and contemplate before making investments in mutual funds. A prospectus serves as a valuable tool for investors to identify the Fund Manager and the mutual funds that will serve as their investment targets.",
    'fundsheet':"This is the fundsheets document {file_name} that issued monthly by the Fund Managers. It furnishes details regarding product performance, asset composition, and the securities portfolio at the end of each month for each mutual fund."
}

deskripsi = "This content contains about prospectus and fund sheet about {document}. Use this tool if you want to answer any questions about {document}.\n"

class MultiDocumentRetriever():

    def __init__(self,):
        self.agents_dict = {}
        self.query_engine = {}
        self.prod_retriever = []
        self.pymupdf = PymuPDF()
        self.all_nodes = []

    def build_document_retriever(self, pdf_folder, nodes_folder, summary_folder, file):

        file_name = str(file.split('.pdf')[0]).lower()

        if file_name.endswith('prospektus'):
            name = file_name.split('_prospektus')[0]
            tipe = 'prospektus'
        else:
            name = file_name.split('_fundsheet')[0]
            tipe = 'fundsheet'

        html_path = f'{pdf_folder}/{file_name}.pdf'
        nodes_path = f'{nodes_folder}/{file_name}.pkl'
        summary_path = f'{summary_folder}/{file_name}.txt'

        base_nodes, summary = self.pymupdf.get_nodes_from_documents(html_path, file_name, nodes_path, summary_path)

        self.all_nodes.extend(base_nodes)

        prompt = prompt_type[tipe].format(file_name=name)

        retriever = VectorStoreIndex(base_nodes).as_retriever(similarity_top_k=10)

        retriever_tool = RetrieverTool.from_defaults(retriever = retriever, 
                                                description = prompt+summary)
        return retriever_tool,name
    
    def build_retriever(self, html_folder, nodes_folder, summary_folder):

        for file in tqdm(os.listdir(html_folder)):
            
            retriever_tool, name = self.build_document_retriever(pdf_folder=html_folder,
                                   nodes_folder=nodes_folder,
                                   summary_folder=summary_folder,
                                   file=file)
            
            if name not in self.agents_dict.keys():
                self.agents_dict[name]=[]
            self.agents_dict[name].append(retriever_tool)
        
        for name in self.agents_dict.keys():

            retriever = RouterRetriever(selector=PydanticMultiSelector.from_defaults(llm=llm),
                                        retriever_tools=self.agents_dict[name],)
            
            retriever = RetrieverTool.from_defaults(retriever = retriever, 
                                                    description = deskripsi.format(document=name))

            
            self.prod_retriever.append(retriever)
        
        keyword_index = SimpleKeywordTableIndex(self.all_nodes)
        self.keyword_retriever = keyword_index.as_retriever(service_context=service_context)
        
        self.prod_retriever.append(RetrieverTool.from_defaults(retriever=self.keyword_retriever,
                                                               description = "Useful for retrieving specific context using keywords"))
        
        self.top_agent = RouterRetriever(selector=PydanticMultiSelector.from_defaults(llm=llm),
                        retriever_tools=self.prod_retriever)
        
        return self.top_agent
    
    def build_recursive_retriever_document(self, raw_nodes, node_mappings):
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
        return recursive_retriever