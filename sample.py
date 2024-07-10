import os
import json
import asyncio
import PyPDF2
import textwrap
import streamlit as st
from dotenv import load_dotenv

from semantic_kernel import Kernel
from semantic_kernel.connectors.ai.open_ai import OpenAIChatCompletion, OpenAIChatPromptExecutionSettings
from semantic_kernel.prompt_template import PromptTemplateConfig, InputVariable
from semantic_kernel.memory.semantic_text_memory import SemanticTextMemory
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.chat_models import ChatOpenAI
from langchain.schema import SystemMessage, HumanMessage, AIMessage

from cdb import upsert_documents, query_collection, document_exists

# Load environment variables
load_dotenv()
os.environ['OPENAI_API_KEY'] = os.getenv("OPENAI_API_KEY")

# Initialize the chat model
chat = ChatOpenAI(
    openai_api_key=os.environ["OPENAI_API_KEY"],
    model='gpt-3.5-turbo'
)

embed_model = OpenAIEmbeddings(model="text-embedding-ada-002")

# Streamlit app
st.title('PDF Question Answering System')

uploaded_files = st.file_uploader("Choose PDF files", type="pdf", accept_multiple_files=True)
#all_chunks =[
#"Thomas James ♂phone+91-9895136620 /envel⌢pe7homasjames@gmail.com Specialised in Machine learning, Data Analytics & Optimization Techniques. Hard-working, Ambitious, Creative & Proactive. Eager to Acquire new Knowledge in order to develop myself as a Strong & Kind Human Being./githubGitHub/7homasjames Experience •VRITIKA Research Intern, Science and Engineering Research Board (SERB) Trivandrum , India Deep Learning and Computer Vision Research Internship under Dr. Sinnu Sussan Thomas 1 month –Acquired extensive expertise in, Bayesian optimization techniques including Gaussian Processes modeling, acqui- sition function design, and surrogate modeling. Proficient in utilizing Bayesian optimization to efficiently explore and exploit the parameter space, accelerating the convergence of optimization processes. Projects –Imagined Speech using EEG Cognitive Computing Project under Dr. Elizabeth Sherly ∗Generated an, Imagined Speech Code Book using EEG signals from the brain in order to model"
#,"new Imagined words for each subject –Hyper Spectral Image Classification Soft Computing Project under Dr. Sujoy Madhab Roy ∗Classification task on Indian Pines HSI dataset using Controlled Random Sampling and Normalized Afiine Hull Distance –Naive Bayes Spam Filter Machine Learning Project ∗Implemented a Spam Filter from Scratch by vectorizing the words and then finding out the probabilities of each word to be either classified as spam or ham Education –Kerala University Of Digital Sciences and Innovation Technology 2022-2024 MSc Computer Science with Specialization in Machine Intelligence CGPA: 7.42/10 –Sacred Heart College, Thevara 2019-2022 Bachelor of Physics CGPA: 8.91/10 Technical Skills Expertise : Research, Machine learning, Deep learning, Reinforcement Learning, Computer Vision, Image processing, Optimization, Soft Computing & Fuzzy Logic, Genetic Algorithm, Fundamentals of Blockchain Technology, Probability & Statistics, Algorithms, Cognitive Computing Languages : Python,"
#,"C++, Arduino, Tiny ML Developer Tools : VS Code, Anaconda, GPT API integrations Frameworks : Tensorflow, Pytorch, Keras, GpyOpt, ScipyOpt Softskills : Critical Thinking, Leadership, Time Management, Teamwork, Communication, Creative thinking, Analytical & Problem solving Certifications –National Cadet Corps Air Wing C Certificate –Microsoft Office Excel Specialist Badge –Python Certificate for Data Science & Machine Learning"
#]

if uploaded_files:
    all_chunks = []
    for uploaded_file in uploaded_files:
        pdf_text = ''
        reader = PyPDF2.PdfReader(uploaded_file)
        for page in reader.pages:
            pdf_text += page.extract_text() or ''
        
        # Process the text into chunks
        chunk_size = 1000
        chunks = textwrap.wrap(pdf_text, chunk_size)
        all_chunks.extend(chunks)
    
    st.write(f"Total extracted text length: {sum(len(chunk) for chunk in all_chunks)} characters")
    st.write(f"Total number of chunks: {len(all_chunks)}")

    # Upsert chunks into ChromaDB and get their IDs
    ids = upsert_documents(documents=all_chunks)
    st.write(f"Successfully inserted {len(ids)} chunks into ChromaDB")

    def augment_prompt(query: str):
        query_results = query_collection(query_texts=[query], n_results=3)
        try:
            source_knowledge = "\n".join([doc for result in query_results['documents'] for doc in result])
        except KeyError as e:
            st.error(f"KeyError: {e}")
            source_knowledge = "No context found due to error."

        augmented_prompt = f"""Using the contexts below, answer the query.
        
        Contexts:
        {source_knowledge}
        
        Query: {query}"""
        return augmented_prompt

    query = st.text_input("Ask a question about the PDF content")

    if query:
        prompt = HumanMessage(content=augment_prompt(query))

        messages = [
            SystemMessage(content="You are a helpful assistant."),
            HumanMessage(content="Hi AI, how are you today?"),
            AIMessage(content="I'm great thank you. How can I help you?"),
            prompt
        ]

        res = chat(messages)
        st.write(res.content)

# Initialize Semantic Kernel
kernel = Kernel()

# Add OpenAI Chat Completion Service
openai_service = OpenAIChatCompletion(
    api_key=os.getenv("OPENAI_API_KEY"),
    ai_model_id="gpt-3.5-turbo",
    service_id=None
)
kernel.add_service(openai_service)
print("Added OpenAI Chat Service...")

# Add OpenAI Embedding Service
embed_model = OpenAIEmbeddings(model="text-embedding-ada-002")

# Setup memory
memory = SemanticTextMemory(storage=all_chunks, embeddings_generator=embed_model)
print("Registered Azure Cosmos DB Memory Store...")

# Prompt Template for Summarization
prompt = """{{$input}}
Summarize the content above."""

execution_settings = OpenAIChatPromptExecutionSettings(
    service_id=None,
    ai_model_id="gpt-3.5-turbo",
    max_tokens=2000,
    temperature=0.7,
)

prompt_template_config = PromptTemplateConfig(
    template=prompt,
    name="summarize",
    template_format="semantic-kernel",
    input_variables=[
        InputVariable(name="input", description="The user input", is_required=True),
    ],
    execution_settings=execution_settings,
)

# Function for Chat Completion with Grounding
prompt = """
    You are a chatbot that can have a conversation about any topic related to the provided context.
    Give explicit answers from the provided context or say 'I don't know' if it does not have an answer.
    Provided context: {{$db_record}}

    User: {{$query_term}}
    Chatbot:"""

chat_execution_settings = OpenAIChatPromptExecutionSettings(
    service_id="chat_completion",
    ai_model_id="gpt-3.5-turbo",
    max_tokens=500,
    temperature=0.0,
    top_p=0.5
)

chat_prompt_template_config = PromptTemplateConfig(
    template=prompt,
    name="grounded_response",
    template_format="semantic-kernel",
    input_variables=[
        InputVariable(name="db_record", description="The database record", is_required=True),
        InputVariable(name="query_term", description="The user input", is_required=True),
    ],
    execution_settings=chat_execution_settings,
)

chat_function = kernel.add_function(
    function_name="ChatGPTFunc",
    plugin_name="chatGPTPlugin",
    prompt_template_config=chat_prompt_template_config
)

# Example Invocation of the Chat Function
query = "What is the capital of France?"  # Example query
result = query_collection(query_texts=[query], n_results=3)  # Example collection query

print(result)
