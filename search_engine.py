import sys
import os
import time
sys.path.append('utils')
import streamlit as st
from utils.retriever import MultiDocumentRetriever

st.title("Your RAGs Search Engine Optimization, powered by LlamaIndex 💬🦙")

if "messages" not in st.session_state.keys():  # Initialize the chat messages history
    st.session_state.messages = [
        {"role": "assistant", "content": "Greetings, I'm here to assist you"}
    ]

@st.cache_resource(show_spinner=False)
def load_data():
    with st.spinner(text="Loading and indexing the Streamlit docs – hang tight! This should take 1-2 minutes."):
        multiDocumentRetriever = MultiDocumentRetriever()
        top_agent_retriever = multiDocumentRetriever.build_retriever(html_folder= 'data/pdf',
                                                             nodes_folder = 'data/nodes',
                                                             summary_folder = 'data/summary',
                                                             model_path = 'model/cache_retriever')
        return top_agent_retriever

top_agent = load_data()

prompt = st.text_input(
            label=":blue[Search your Query]",
            placeholder="Your query...",)

search_button = st.button(label="Run", type="primary")

if prompt:
    # And if they have clicked the search button
    if search_button:
        response = top_agent.retrieve(prompt)
        for text in response:
            output = text.text
            output = output.replace('\n',' ').split()
            st.text_area(label=f"{text.node_id}/page-{text.metadata['page_number']}",value=' '.join(output[:50])+'...')