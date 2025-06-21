import gradio as gr
import requests

# Global chat history
chat_history = []

def chat_with_ollama(user_input):
    global chat_history
    chat_history.append({"role": "user", "content": user_input})

    try:
        response = requests.post(
            "http://localhost:11434/api/chat",
            json={
                "model": "llama3",  # Replace with "llama3.2" if you still want the older version
                "messages": chat_history,
                "stream": False
            }
        )
        reply = response.json()["message"]["content"]
    except Exception as e:
        reply = f"⚠️ Error: {str(e)}"

    chat_history.append({"role": "assistant", "content": reply})

    # Format for Gradio's chatbot display
    chat_display = [(m["content"], "") if m["role"] == "user" else ("", m["content"]) for m in chat_history]
    return "", chat_display

# Gradio UI
with gr.Blocks() as demo:
    gr.Markdown("## 🤖 Chat with Ollama (LLaMA 3)")
    chatbot = gr.Chatbot()
    msg = gr.Textbox(label="Your Message")
    msg.submit(chat_with_ollama, inputs=msg, outputs=[msg, chatbot])

# Run the app
demo.launch()
