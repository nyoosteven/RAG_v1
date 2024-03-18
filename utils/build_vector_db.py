import fitz  # PyMuPDF
import os
import pickle
import nest_asyncio
import nltk
import json

from llama_index.core import (VectorStoreIndex,
                         ServiceContext,
                         StorageContext,
                         SimpleDirectoryReader,
                         SummaryIndex,
                         set_global_service_context)
from llama_index.core.schema import TextNode, NodeRelationship, RelatedNodeInfo
from llama_index.llms.openai import OpenAI
from llama_index.core.callbacks import CallbackManager, LlamaDebugHandler
from llama_index.core.node_parser import SentenceSplitter

from pathlib import Path
nest_asyncio.apply()
llama_debug = LlamaDebugHandler(print_trace_on_end=True)
callback_manager = CallbackManager([llama_debug])
os.environ["OPENAI_API_KEY"] = ""

nltk.data.path.append('nltk_data')

llm = OpenAI(temperature=0, 
             model="gpt-3.5-turbo",
             callback_manager= callback_manager,
             max_token = 1000)

def write_text(path, text):
    with open(path, "w+") as file:
        file.write(text)

def load_text(path):
    with open(path, "r") as file:
        text = file.read()
    return text

class PymuPDF():

    def __init__(self):
        self.text_parser = SentenceSplitter(
            chunk_size=256,
            )
        
    def get_nodes_from_documents(self, file_name, product_name, node_save_path, summary_path):

        if 'pdf' in file_name:
            if node_save_path is None or not os.path.exists(node_save_path):
                nodes = [] 
                docs = fitz.open(file_name)
                for num, page in enumerate(docs):
                    text = page.get_text("text")
                    nodes_tmp = self.text_parser.split_text(text)
                    for node in nodes_tmp:
                        node = TextNode(text = node)
                        node.metadata = {"page_number": num + 1}
                        nodes.append(node)
                    
                for i in range(1, len(nodes)):
                    nodes[i - 1].relationships[NodeRelationship.NEXT] = RelatedNodeInfo(node_id=nodes[i].node_id)
                    nodes[i].relationships[NodeRelationship.PREVIOUS] = RelatedNodeInfo(node_id=nodes[i - 1].node_id)
                
                for node in nodes:
                    node.metadata.update({'file_name':product_name})
                
                now = 1
                node_tmp_mapping={}
                for node in nodes:
                    node_tmp_mapping[node.node_id]=f"{product_name}-node-{now}"
                    now+=1
                    
                for node in nodes:
                    node.id_ = node_tmp_mapping[node.id_]
                    for relationship in node.relationships.values():
                        try:
                            relationship.node_id = node_tmp_mapping[relationship.node_id]
                        except:
                            pass
                
                summary = self.get_summary(nodes)

                pickle.dump(nodes, open(node_save_path, 'wb'))
                write_text(summary_path, summary)

            else:
                nodes = pickle.load(open(node_save_path, 'rb'))
                summary = load_text(summary_path)

        return nodes, summary
    
    def get_summary(self,raw_nodes):

        summary_index = SummaryIndex(raw_nodes) 
        summary_query_engine = summary_index.as_query_engine(response_mode="tree_summarize", llm=llm)
        summary = str(summary_query_engine.query("Extract a concise 1-2 line summary of this document"))
        return summary