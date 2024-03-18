import os
import sys
import json
import copy
import openai
import pickle
import nltk
import nest_asyncio
from typing import Optional
from tqdm import tqdm
nest_asyncio.apply()
from llama_index.core import ServiceContext, download_loader

from llama_index.core.query_engine import RetrieverQueryEngine
from llama_index.llms.openai import OpenAI
from llama_index.core.node_parser import UnstructuredElementNodeParser
from llama_index.core.schema import IndexNode
from llama_index.core.retrievers import RecursiveRetriever
from llama_index.core.callbacks import CallbackManager, LlamaDebugHandler
from llama_index.readers.file.pymu_pdf import PyMuPDFReader
from pathlib import Path

from llama_index.core.evaluation import (
    DatasetGenerator,
    FaithfulnessEvaluator,
    RelevancyEvaluator
)
import openai
import time

llama_debug = LlamaDebugHandler(print_trace_on_end=True)
callback_manager = CallbackManager([llama_debug])
os.environ["OPENAI_API_KEY"] = ""

nltk.data.path.append('nltk_data')

llm = OpenAI(temperature=0, 
             model="gpt-3.5-turbo",
             callback_manager= callback_manager,
             max_token = 1000)

service_context_gpt4 = ServiceContext.from_defaults(llm=llm)

# Define Faithfulness and Relevancy Evaluators which are based on GPT-4
#faithfulness_gpt4 = FaithfulnessEvaluator(service_context=service_context_gpt4)
#relevancy_gpt4 = RelevancyEvaluator(service_context=service_context_gpt4)
loader = PyMuPDFReader()
questions = []
for file_name in tqdm(os.listdir('data/pdf')):
    documents = loader.load(f'data/pdf/{file_name}')

    file_name = str(file_name.split('.pdf')[0]).lower()
    if file_name.endswith('prospektus'):
        name = file_name.split('_prospektus')[0]
    else:
        name = file_name.split('_fundsheet')[0]
    name = name.replace("_"," ")
    data_generator = DatasetGenerator.from_documents(documents)
    eval_questions = data_generator.generate_questions_from_nodes(num=3)
    for question in eval_questions:
        question = question[:-1]
        questions.append(question+f' about {name} ?')

with open('data/question.txt', "w+") as file:
    for question in questions:
        file.write(question+'\n')
