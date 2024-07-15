# Semantic_Kernel
Semantic Kernel (SK) is an open-source framework developed by Microsoft, designed to integrate AI models with traditional programming languages, providing a bridge between natural language processing (NLP) and software development. The framework is particularly useful for building and deploying AI-powered applications, allowing developers to leverage the capabilities of large language models like GPT-4 within their own applications.
![image](https://github.com/user-attachments/assets/6afb9678-ef0e-4b67-b32f-31aa6ad93846)


## Into the Code

The Streamlit application is demonstrated in `app.py`, which converts the RAG model built using semantic kernel into a Streamlit application. This application allows users to upload a PDF file and generate responses based on the content of the PDF using the RAG model.

## How to Use the Streamlit Application

### Prerequisites

- Python 3.6 or higher
- `pip` for installing Python packages

### Installation

1. Clone the repository:
   ```bash
    git clone https://github.com/7homasjames/Semantic_Kernel.git
    ```
2. Create a virtual Enviornmnet:
   ```bash
    python -m venv venv

    ```

3. Install the required packages:

    ```bash
    pip install -r requirements.txt
    ```

### Running the Example


1. Prepare your `.env` file with your API keys:

    ```plaintext
    OPENAI_API_KEY=your_openai_api_key
    ```

2. Run the main script:

    ```bash
    python -m streamlit run app.py   
    ```
