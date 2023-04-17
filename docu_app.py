import pdfminer
import pdfminer.high_level
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import FAISS
from langchain.chains.question_answering import load_qa_chain
from langchain.chat_models import ChatOpenAI
from langchain.chains import ConversationalRetrievalChain
import pickle
from pathlib import Path
from dotenv import load_dotenv
import os
import streamlit as st
from streamlit_chat import message
import io
import asyncio
from docx import Document

load_dotenv()
api_key = os.getenv('OPENAI_API_KEY')  

async def main():        

    async def storeDocEmbeds(files,filename):
        corpus = "\n".join([pdfminer.high_level.extract_text(io.BytesIO(uploaded_file.read())) if uploaded_file.name.split(".")[-1].lower() == "pdf" else "\n".join([para.text.strip() for para in Document(io.BytesIO(uploaded_file.read())).paragraphs]) if uploaded_file.name.split(".")[-1].lower() in ["docx", "doc"] else uploaded_file.read().decode('utf-8') if uploaded_file.name.split(".")[-1].lower() == "txt" else "" for uploaded_file in files if uploaded_file])
        splitter =  RecursiveCharacterTextSplitter(chunk_size=1000,chunk_overlap=200,)
        chunks = splitter.split_text(corpus)
        
        #For use with Openai embeddings
        embeddings = OpenAIEmbeddings(openai_api_key = api_key)
        vectors = FAISS.from_texts(chunks, embeddings)

        #for use with openAI embeddings
        with open(filename + ".pkl", "wb") as f:
            pickle.dump(vectors, f)
            print("embeddings successfully stored")  
        
    async def getDocEmbeds(file, filename):
        
        if not os.path.isfile(filename + ".pkl"):
            await storeDocEmbeds(file, filename)
        
        with open(filename + ".pkl", "rb") as f:
            global vectores
            vectors = pickle.load(f)
        
        return vectors

    async def conversational_chat(query):
        result = qa({"question": query, "chat_history": st.session_state['history']})
        st.session_state['history'].append((query, result["answer"]))
        return result["answer"]

    llm = ChatOpenAI(model_name="gpt-3.5-turbo")
    chain = load_qa_chain(llm, chain_type="stuff")

    if 'history' not in st.session_state:
        st.session_state['history'] = []

    #Creating the chatbot interface
    st.header("DocuChatter")
    st.subheader("A conversational chatbot for summarizing and answering questions about documents")

    if 'ready' not in st.session_state:
        st.session_state['ready'] = False

    uploaded_files = st.file_uploader("Upload your files", accept_multiple_files = True, type=['pdf','doc','docx','txt'])

    if len(uploaded_files) >0:

        with st.spinner("Processing..."):
        # Add your code here that needs to be executed
            vectors = await getDocEmbeds(uploaded_files,uploaded_files[0].name)
            qa = ConversationalRetrievalChain.from_llm(ChatOpenAI(model_name="gpt-3.5-turbo"), retriever=vectors.as_retriever(), return_source_documents=True)

        st.session_state['ready'] = True

    st.divider()

    if st.session_state['ready']:

        if 'generated' not in st.session_state:
            st.session_state['generated'] = ["Welcome! You can now ask any questions regarding your files"]

        if 'past' not in st.session_state:
            st.session_state['past'] = ["Hi Docuchatter!"]

        # container for chat history
        response_container = st.container()

        # container for text box
        container = st.container()

        with container:
            with st.form(key='my_form', clear_on_submit=True):
                user_input = st.text_input("Query:", placeholder="e.g: Summarize the files in a few sentences", key='input')
                submit_button = st.form_submit_button(label='Send')

            if submit_button and user_input:
                output = await conversational_chat(user_input)
                st.session_state['past'].append(user_input)
                st.session_state['generated'].append(output)

        if st.session_state['generated']:
            with response_container:
                for i in range(len(st.session_state['generated'])):
                    message(st.session_state["past"][i], is_user=True, key=str(i) + '_user', avatar_style="thumbs")
                    message(st.session_state["generated"][i], key=str(i), avatar_style="adventurer", seed=123)


if __name__ == "__main__":
    asyncio.run(main())