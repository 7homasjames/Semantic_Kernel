import os
import textwrap
import streamlit as st
from dotenv import load_dotenv
import PyPDF2
import asyncio
from semantic_kernel.functions import KernelArguments
from semantic_kernel import Kernel
from semantic_kernel.connectors.ai.open_ai import (
    AzureChatCompletion,
    AzureTextEmbedding,
    AzureChatPromptExecutionSettings
)
from semantic_kernel.prompt_template import PromptTemplateConfig, InputVariable
from semantic_kernel.memory import SemanticTextMemory, VolatileMemoryStore
from semantic_kernel.core_plugins.text_memory_plugin import TextMemoryPlugin

# Load environment variables
load_dotenv()
os.environ['AZURE_OPENAI_API_KEY'] = os.getenv("AZURE_OPENAI_API_KEY")
os.environ['AZURE_OPENAI_ENDPOINT'] = os.getenv("AZURE_OPENAI_ENDPOINT")
os.environ['AZURE_OPENAI_DEPLOYMENT_NAME'] = os.getenv("AZURE_OPENAI_DEPLOYMENT_NAME")
os.environ['AZURE_OPENAI_EMBEDDING_DEPLOYMENT_NAME'] = os.getenv("AZURE_OPENAI_EMBEDDING_DEPLOYMENT_NAME")

# Streamlit app
st.title('Semantic Kernel Question Answering System💬')

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
    
    st.write(f"Total number of chunks: {len(all_chunks)}")

    query = st.text_input("Type HI to Initialize the Semantic Kernel and then Ask question")

    async def get_response(query):
        # Initialize Semantic Kernel
        kernel = Kernel()

        # Add Azure OpenAI Chat Completion Service
        azure_openai_service = AzureChatCompletion(
            service_id="chat_completion",
            api_key=os.getenv("AZURE_OPENAI_API_KEY"),
            endpoint=os.getenv("AZURE_OPENAI_ENDPOINT"),
            deployment_name=os.getenv("AZURE_OPENAI_DEPLOYMENT_NAME")
        )
        kernel.add_service(azure_openai_service)

        azure_openai_embedding_service = AzureTextEmbedding(
            service_id="text_embedding",
            api_key=os.getenv("AZURE_OPENAI_API_KEY"),
            endpoint=os.getenv("AZURE_OPENAI_ENDPOINT"),
            deployment_name=os.getenv("AZURE_OPENAI_EMBEDDING_DEPLOYMENT_NAME")
        )
        kernel.add_service(azure_openai_embedding_service)

        # Setup memory
        memory = SemanticTextMemory(storage=VolatileMemoryStore(), embeddings_generator=azure_openai_embedding_service)
        kernel.add_plugin(TextMemoryPlugin(memory), "TextMemoryPluginACDB")

        # Prompt Template for Chat Completion with Grounding
        prompt_template = """
            You are a chatbot that can have a conversation about any topic related to the provided context.
            Give explicit answers from the provided context or say 'I don't know' if it does not have an answer.
            Provided context: {{$db_record}}

            User: {{$query_term}}
            Chatbot:"""

        chat_execution_settings = AzureChatPromptExecutionSettings(
            deployment_name=os.getenv("AZURE_OPENAI_DEPLOYMENT_NAME"),
            max_tokens=1000,
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

        arguments = KernelArguments(db_record="\n".join(all_chunks), query_term=query)

        result = await kernel.invoke(
            chat_function,arguments
        )

        st.write(result)
        return result

    if query:
        asyncio.run(get_response(query))
