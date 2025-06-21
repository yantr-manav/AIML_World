import requests
from colorama import init, Fore, Style

# Initialize colorama
init(autoreset=True)

# Chat history to maintain context
chat_history = []

print(Fore.YELLOW + "🧠 Terminal Chat with Ollama (LLaMA 3)\n" + Style.RESET_ALL)

while True:
    try:
        # User input
        user_input = input(Fore.CYAN + "You: " + Style.RESET_ALL)
        if user_input.lower() in ['exit', 'quit']:
            print(Fore.YELLOW + "👋 Exiting chat. Goodbye!")
            break

        chat_history.append({"role": "user", "content": user_input})

        # Call Ollama API
        response = requests.post(
            "http://localhost:11434/api/chat",
            json={
                "model": "llama3",
                "messages": chat_history,
                "stream": False
            }
        )

        # Extract reply
        reply = response.json()["message"]["content"]
        chat_history.append({"role": "assistant", "content": reply})

        # Print assistant's reply
        print(Fore.GREEN + "Ollama: " + Style.RESET_ALL + reply)

    except Exception as e:
        print(Fore.RED + f"⚠️ Error: {str(e)}")
