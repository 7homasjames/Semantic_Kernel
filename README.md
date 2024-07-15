# Semantic_Kernel
Semantic Kernel is a framework developed by Microsoft that facilitates the integration of Large Language Models (LLMs) with conventional programming languages
![image](https://github.com/user-attachments/assets/9748b258-74e6-469e-8e09-9fbb25378ed3)

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

2. Install the required packages:

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
