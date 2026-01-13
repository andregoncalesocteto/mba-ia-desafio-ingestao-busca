from search import search_prompt

def main():
    """
    Interactive CLI chat interface for RAG-based question answering.

    The chat loop:
    1. Initializes the RAG chain using search_prompt()
    2. Prompts user for questions
    3. Retrieves relevant context from vector database
    4. Generates responses using LLM
    5. Continues until user exits
    """
    # Initialize the RAG chain
    print("Inicializando o sistema de busca...")
    chain = search_prompt()

    if not chain:
        print("Não foi possível iniciar o chat. Verifique os erros de inicialização.")
        return

    # Welcome message
    print("\n" + "="*60)
    print("Chat RAG - Sistema de Busca com IA")
    print("="*60)
    print("Digite sua pergunta e pressione Enter.")
    print("Para sair, digite: 'quit', 'exit' ou 'sair'")
    print("="*60 + "\n")

    # Chat loop
    while True:
        try:
            # Get user input
            pergunta = input("Pergunta: ").strip()

            # Check for exit commands
            if pergunta.lower() in ["quit", "exit", "sair", ""]:
                print("\nEncerrando o chat. Até logo!")
                break

            # Invoke chain with user question
            print("\nBuscando informações...\n")
            resposta = chain.invoke(pergunta)

            # Display response
            print(f"Resposta: {resposta}\n")
            print("-" * 60 + "\n")

        except KeyboardInterrupt:
            print("\n\nChat interrompido pelo usuário. Até logo!")
            break
        except Exception as e:
            print(f"\nErro ao processar pergunta: {e}\n")
            print("Tente novamente ou digite 'sair' para encerrar.\n")

if __name__ == "__main__":
    main()