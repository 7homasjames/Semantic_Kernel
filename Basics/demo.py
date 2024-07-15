import os
import textwrap
import streamlit as st
from dotenv import load_dotenv
import PyPDF2
import asyncio

from semantic_kernel import Kernel
from semantic_kernel.connectors.ai.open_ai import OpenAIChatCompletion, OpenAIChatPromptExecutionSettings, OpenAITextEmbedding
from semantic_kernel.prompt_template import PromptTemplateConfig, InputVariable
from semantic_kernel.memory.semantic_text_memory import SemanticTextMemory
from semantic_kernel.core_plugins.text_memory_plugin import TextMemoryPlugin

from cdb import upsert_documents, query_collection

# Load environment variables
load_dotenv()
os.environ['OPENAI_API_KEY'] = os.getenv("OPENAI_API_KEY")


# Streamlit app
st.title('PDF Question Answering System')

uploaded_files = st.file_uploader("Choose PDF files", type="pdf", accept_multiple_files=True)

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

    async def get_response(query):
        prompt = augment_prompt(query)

        # Initialize Semantic Kernel
        kernel = Kernel()

        # Add OpenAI Chat Completion Service
        openai_service = OpenAIChatCompletion(
            api_key=os.getenv("OPENAI_API_KEY"),
            ai_model_id="gpt-3.5-turbo"
        )
        kernel.add_service(openai_service)

        openai_embedding_service = OpenAITextEmbedding(
            api_key=os.getenv("OPENAI_API_KEY"),
            ai_model_id="text-embedding-ada-002"
        )
        kernel.add_service(openai_embedding_service)

        # Setup memory
        memory = SemanticTextMemory(storage=all_chunks, embeddings_generator=openai_embedding_service)
        kernel.add_plugin(TextMemoryPlugin(memory), "TextMemoryPluginACDB")

        # Prompt Template for Chat Completion with Grounding
        prompt_template = """
            You are a chatbot that can have a conversation about any topic related to the provided context.
            Give explicit answers from the provided context or say 'I don't know' if it does not have an answer.
            Provided context: {{$db_record}}

            User: {{$query_term}}
            Chatbot:"""

        chat_execution_settings = OpenAIChatPromptExecutionSettings(
            ai_model_id="gpt-3.5-turbo",
            max_tokens=500,
            temperature=0.0,
            top_p=0.5
        )

        chat_prompt_template_config = PromptTemplateConfig(
            template=prompt_template,
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

        
        #result = chat_function.invoke({
        #   "db_record": "\n".join(all_chunks),
        #    "query_term": query
        #})

        #st.write(result)
        result=await memory.search(collection="\n".join(all_chunks), query= query)
        st.write(print(result[0].text))
        return result
       

    if query:
        response = asyncio.run(get_response(query))
        st.write(response)
