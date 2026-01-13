import os
from dotenv import load_dotenv, find_dotenv
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_postgres import PGVector
from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser

# Load environment variables
load_dotenv(find_dotenv(), override=True)

PROMPT_TEMPLATE = """
CONTEXTO:
{contexto}

REGRAS:
- Responda somente com base no CONTEXTO.
- Se a informação não estiver explicitamente no CONTEXTO, responda:
  "Não tenho informações necessárias para responder sua pergunta."
- Nunca invente ou use conhecimento externo.
- Nunca produza opiniões ou interpretações além do que está escrito.

EXEMPLOS DE PERGUNTAS FORA DO CONTEXTO:
Pergunta: "Qual é a capital da França?"
Resposta: "Não tenho informações necessárias para responder sua pergunta."

Pergunta: "Quantos clientes temos em 2024?"
Resposta: "Não tenho informações necessárias para responder sua pergunta."

Pergunta: "Você acha isso bom ou ruim?"
Resposta: "Não tenho informações necessárias para responder sua pergunta."

PERGUNTA DO USUÁRIO:
{pergunta}

RESPONDA A "PERGUNTA DO USUÁRIO"
"""

def search_prompt(question=None):
    """
    Initialize and return a RAG chain for question answering.

    Steps:
    1. Vectorize the question using OpenAI embeddings
    2. Search for k=10 most relevant chunks using similarity_search_with_score
    3. Format the prompt with retrieved context
    4. Call LLM to generate response

    Returns:
        A LangChain chain that can be invoked with {"pergunta": "user question"}
        Returns None if initialization fails.
    """
    try:
        # Validate required environment variables
        database_url = os.getenv("DATABASE_URL")
        if not database_url:
            print("Erro: DATABASE_URL não configurado.")
            return None

        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            print("Erro: OPENAI_API_KEY não configurado.")
            return None

        embedding_model = os.getenv("OPENAI_EMBEDDING_MODEL", "text-embedding-3-small")
        collection_name = os.getenv("PG_VECTOR_COLLECTION_NAME", "documents")

        # Initialize embeddings (same model used during ingestion)
        embeddings = OpenAIEmbeddings(
            model=embedding_model,
            api_key=api_key,
        )

        # Connect to PGVector store
        vectorstore = PGVector(
            connection=database_url,
            collection_name=collection_name,
            embeddings=embeddings,
        )

        # Create retriever with k=10 for similarity search
        retriever = vectorstore.as_retriever(
            search_type="similarity",
            search_kwargs={"k": 10}
        )

        # Initialize LLM
        llm = ChatOpenAI(
            model="gpt-4o-mini",
            temperature=0,
            api_key=api_key,
        )

        # Create prompt template
        prompt = PromptTemplate(
            template=PROMPT_TEMPLATE,
            input_variables=["contexto", "pergunta"]
        )

        # Helper function to format retrieved documents into context
        def format_docs(docs):
            return "\n\n".join(doc.page_content for doc in docs)

        # Build RAG chain using LCEL
        chain = (
            {
                "contexto": retriever | format_docs,
                "pergunta": RunnablePassthrough()
            }
            | prompt
            | llm
            | StrOutputParser()
        )

        return chain

    except Exception as e:
        print(f"Erro ao inicializar o sistema de busca: {e}")
        return None