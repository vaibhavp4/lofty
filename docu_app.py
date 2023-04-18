import pdfminer
import pdfminer.high_level
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import FAISS
from langchain.chains.question_answering import load_qa_chain
import openai
import pickle
from pathlib import Path
from dotenv import load_dotenv
import os
import io
import asyncio
from docx import Document

load_dotenv()
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

    async def conversational_chat(query, uploaded_files, openai_api_key):
        os.env.set("OPENAI_API_KEY", openai_api_key)
        vectors = await getDocEmbeds(uploaded_files,uploaded_files[0].name)
        chain = load_qa_chain(OpenAI(temperature=0), chain_type="stuff")
        result = chain.run(input_documents=vectors, question=query)
        return result["output_text"]

    