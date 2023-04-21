BabyAGI is a task driven autonomous agent designed to create and complete a task list based on a given goal. Here, we use this framework to analyze a given file rigorously by asking questions and prioritising them. The flow chart below provides an implementation of BabyAGI in combination with file analysis and question-answering capabilities using various libraries, such as pdfminer, docx, and Streamlit.

**File Processing**
To process different file types first we extract text from PDF, DOCX, and TXT files
Using Langchain splitters, Chromadb, and openAI embeddings, we create a retriever from the text
Again using Langchain, we initialise a chain based on the retreiver which can then summarise insights from the file based on a given query. This chain is leveraged by the Retrieval chain that replaces the execution agent of BabyAGI

**BabyAGI Components**
The BabyAGI implementation consists of several components, each responsible for a specific functionality:
- Question Creation Chain: Generates new research questions based on previous results.
- Question Prioritisation Chain: Cleans up and prioritizes the list of generated research questions.
- Retrieval Chain: Generates new insights based on the context from the uploaded file and question under investigation.
BabyAGI Controller: Manages the entire process, adding and executing tasks while ensuring that the generated insights are stored for future reference.

**Streamlit Interface**
The implementation leverages the Streamlit library to create a user-friendly interface for uploading files and running the BabyAGI AI research assistant. Users can provide their OpenAI API Key, upload a file, specify the maximum number of iterations, and initiate the AI-powered file analysis by clicking the “Run” button.

**Conclusion**
The provided code snippet offers a robust implementation of BabyAGI, as a research assistant on uploaded files that can analyse various file types and generate insights by asking questions and prioritising them. This implementation can be easily extended and customised to fit specific use cases or integrate with other systems.
Some use cases could include topic modelling or combining the agent with internet access, filling the missing piece of information from online sources.

**Credits:**
- https://github.com/yoheinakajima/babyagi
- https://github.com/dory111111/babyagi-streamlit
- https://python.langchain.com/en/latest/
