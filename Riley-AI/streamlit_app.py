import torch
import gradio as gr
from transformers import pipeline

import gradio as gr
from models import ask_riley
from riley_genesis import RileyCore
import tempfile
from TTS.api import TTS

import gradio as gr
import os
api_token = os.getenv("HF_TOKEN")


from langchain_community.vectorstores import FAISS
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain.chains import ConversationalRetrievalChain
from langchain_community.embeddings import HuggingFaceEmbeddings 
from langchain_community.llms import HuggingFacePipeline
from langchain.chains import ConversationChain
from langchain.memory import ConversationBufferMemory
from langchain_community.llms import HuggingFaceEndpoint
import torch

list_llm = ["meta-llama/Meta-Llama-3-8B-Instruct", "mistralai/Mistral-7B-Instruct-v0.2"]  
list_llm_simple = [os.path.basename(llm) for llm in list_llm]

# Load and split PDF document
def load_doc(list_file_path):
    # Processing for one document only
    # loader = PyPDFLoader(file_path)
    # pages = loader.load()
    loaders = [PyPDFLoader(x) for x in list_file_path]
    pages = []
    for loader in loaders:
        pages.extend(loader.load())
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size = 1024, 
        chunk_overlap = 64 
    )  
    doc_splits = text_splitter.split_documents(pages)
    return doc_splits

# Create vector database
def create_db(splits):
    embeddings = HuggingFaceEmbeddings()
    vectordb = FAISS.from_documents(splits, embeddings)
    return vectordb


# Initialize langchain LLM chain
def initialize_llmchain(llm_model, temperature, max_tokens, top_k, vector_db, progress=gr.Progress()):
    if llm_model == "meta-llama/Meta-Llama-3-8B-Instruct":
        llm = HuggingFaceEndpoint(
            repo_id=llm_model,
            huggingfacehub_api_token = api_token,
            temperature = temperature,
            max_new_tokens = max_tokens,
            top_k = top_k,
        )
    else:
        llm = HuggingFaceEndpoint(
            huggingfacehub_api_token = api_token,
            repo_id=llm_model, 
            temperature = temperature,
            max_new_tokens = max_tokens,
            top_k = top_k,
        )
    
    memory = ConversationBufferMemory(
        memory_key="chat_history",
        output_key='answer',
        return_messages=True
    )

    retriever=vector_db.as_retriever()
    qa_chain = ConversationalRetrievalChain.from_llm(
        llm,
        retriever=retriever,
        chain_type="stuff", 
        memory=memory,
        return_source_documents=True,
        verbose=False,
    )
    return qa_chain

# Initialize database
def initialize_database(list_file_obj, progress=gr.Progress()):
    # Create a list of documents (when valid)
    list_file_path = [x.name for x in list_file_obj if x is not None]
    # Load document and create splits
    doc_splits = load_doc(list_file_path)
    # Create or load vector database
    vector_db = create_db(doc_splits)
    return vector_db, "Database created!"

# Initialize LLM
def initialize_LLM(llm_option, llm_temperature, max_tokens, top_k, vector_db, progress=gr.Progress()):
    # print("llm_option",llm_option)
    llm_name = list_llm[llm_option]
    print("llm_name: ",llm_name)
    qa_chain = initialize_llmchain(llm_name, llm_temperature, max_tokens, top_k, vector_db, progress)
    return qa_chain, "QA chain initialized. Chatbot is ready!"


def format_chat_history(message, chat_history):
    formatted_chat_history = []
    for user_message, bot_message in chat_history:
        formatted_chat_history.append(f"User: {user_message}")
        formatted_chat_history.append(f"Assistant: {bot_message}")
    return formatted_chat_history
    

def conversation(qa_chain, message, history):
    formatted_chat_history = format_chat_history(message, history)
    # Generate response using QA chain
    response = qa_chain.invoke({"question": message, "chat_history": formatted_chat_history})
    response_answer = response["answer"]
    if response_answer.find("Helpful Answer:") != -1:
        response_answer = response_answer.split("Helpful Answer:")[-1]
    response_sources = response["source_documents"]
    response_source1 = response_sources[0].page_content.strip()
    response_source2 = response_sources[1].page_content.strip()
    response_source3 = response_sources[2].page_content.strip()
    # Langchain sources are zero-based
    response_source1_page = response_sources[0].metadata["page"] + 1
    response_source2_page = response_sources[1].metadata["page"] + 1
    response_source3_page = response_sources[2].metadata["page"] + 1
    # Append user message and response to chat history
    new_history = history + [(message, response_answer)]
    return qa_chain, gr.update(value=""), new_history, response_source1, response_source1_page, response_source2, response_source2_page, response_source3, response_source3_page
    

def upload_file(file_obj):
    list_file_path = []
    for idx, file in enumerate(file_obj):
        file_path = file_obj.name
        list_file_path.append(file_path)
    return list_file_path


def demo():
    # with gr.Blocks(theme=gr.themes.Default(primary_hue="sky")) as demo:
    with gr.Blocks(theme=gr.themes.Default(primary_hue="red", secondary_hue="pink", neutral_hue = "sky")) as demo:
        vector_db = gr.State()
        qa_chain = gr.State()
        gr.HTML("<center><h1>RAG PDF chatbot</h1><center>")
        gr.Markdown("""<b>Query your PDF documents!</b> This AI agent is designed to perform retrieval augmented generation (RAG) on PDF documents. The app is hosted on Hugging Face Hub for the sole purpose of demonstration. \
        <b>Please do not upload confidential documents.</b>
        """)
        with gr.Row():
            with gr.Column(scale = 86):
                gr.Markdown("<b>Step 1 - Upload PDF documents and Initialize RAG pipeline</b>")
                with gr.Row():
                    document = gr.Files(height=300, file_count="multiple", file_types=["pdf"], interactive=True, label="Upload PDF documents")
                with gr.Row():
                    db_btn = gr.Button("Create vector database")
                with gr.Row():
                        db_progress = gr.Textbox(value="Not initialized", show_label=False) # label="Vector database status", 
                gr.Markdown("<style>body { font-size: 16px; }</style><b>Select Large Language Model (LLM) and input parameters</b>")
                with gr.Row():
                    llm_btn = gr.Radio(list_llm_simple, label="Available LLMs", value = list_llm_simple[0], type="index") # info="Select LLM", show_label=False
                with gr.Row():
                    with gr.Accordion("LLM input parameters", open=False):
                        with gr.Row():
                            slider_temperature = gr.Slider(minimum = 0.01, maximum = 1.0, value=0.5, step=0.1, label="Temperature", info="Controls randomness in token generation", interactive=True)
                        with gr.Row():
                            slider_maxtokens = gr.Slider(minimum = 128, maximum = 9192, value=4096, step=128, label="Max New Tokens", info="Maximum number of tokens to be generated",interactive=True)
                        with gr.Row():
                                slider_topk = gr.Slider(minimum = 1, maximum = 10, value=3, step=1, label="top-k", info="Number of tokens to select the next token from", interactive=True)
                with gr.Row():
                    qachain_btn = gr.Button("Initialize Question Answering Chatbot")
                with gr.Row():
                        llm_progress = gr.Textbox(value="Not initialized", show_label=False) # label="Chatbot status", 

            with gr.Column(scale = 200):
                gr.Markdown("<b>Step 2 - Chat with your Document</b>")
                chatbot = gr.Chatbot(height=505)
                with gr.Accordion("Relevent context from the source document", open=False):
                    with gr.Row():
                        doc_source1 = gr.Textbox(label="Reference 1", lines=2, container=True, scale=20)
                        source1_page = gr.Number(label="Page", scale=1)
                    with gr.Row():
                        doc_source2 = gr.Textbox(label="Reference 2", lines=2, container=True, scale=20)
                        source2_page = gr.Number(label="Page", scale=1)
                    with gr.Row():
                        doc_source3 = gr.Textbox(label="Reference 3", lines=2, container=True, scale=20)
                        source3_page = gr.Number(label="Page", scale=1)
                with gr.Row():
                    msg = gr.Textbox(placeholder="Ask a question", container=True)
                with gr.Row():
                    submit_btn = gr.Button("Submit")
                    clear_btn = gr.ClearButton([msg, chatbot], value="Clear")
            
        # Preprocessing events
        db_btn.click(initialize_database, \
            inputs=[document], \
            outputs=[vector_db, db_progress])
        qachain_btn.click(initialize_LLM, \
            inputs=[llm_btn, slider_temperature, slider_maxtokens, slider_topk, vector_db], \
            outputs=[qa_chain, llm_progress]).then(lambda:[None,"",0,"",0,"",0], \
            inputs=None, \
            outputs=[chatbot, doc_source1, source1_page, doc_source2, source2_page, doc_source3, source3_page], \
            queue=False)

        # Chatbot events
        msg.submit(conversation, \
            inputs=[qa_chain, msg, chatbot], \
            outputs=[qa_chain, msg, chatbot, doc_source1, source1_page, doc_source2, source2_page, doc_source3, source3_page], \
            queue=False)
        submit_btn.click(conversation, \
            inputs=[qa_chain, msg, chatbot], \
            outputs=[qa_chain, msg, chatbot, doc_source1, source1_page, doc_source2, source2_page, doc_source3, source3_page], \
            queue=False)
        clear_btn.click(lambda:[None,"",0,"",0,"",0], \
            inputs=None, \
            outputs=[chatbot, doc_source1, source1_page, doc_source2, source2_page, doc_source3, source3_page], \
            queue=False)
    demo.queue().launch(debug=True)


if __name__ == "__main__":
    demo()
riley = RileyCore()
tts = TTS(model_name="tts_models/en/ljspeech/tacotron2-DDC", progress_bar=False)

def chat_interface(history, user_input):
    if user_input.startswith("!mode"):
        _, mode = user_input.split()
        return history + [{"role": "system", "content": riley.set_mode(mode)}], "", None, history

    if user_input.startswith("!personality"):
        _, profile = user_input.split()
        return history + [{"role": "system", "content": riley.set_personality(profile)}], "", None, history

    context_prompt = riley.think(user_input)
    response_raw = ask_riley(context_prompt)
    response = response_raw.replace('\n', ' ').replace('\r', '').replace('\\', '').strip()

    if "User:" in response:
        response = response.split("User:")[0].strip()

    riley.remember(f"Riley: {response}")
    history.append({"role": "user", "content": user_input})
    history.append({"role": "assistant", "content": response})

    audio_path = tempfile.mktemp(suffix=".wav")
    tts.tts_to_file(text=response, file_path=audio_path)

    return history, "", audio_path, history

css = """
body { background: #0b0f1e; color: #00ffff; font-family: 'Orbitron', sans-serif; }
.gradio-container {
    border: 2px solid #ffaa00; background: linear-gradient(145deg, #000000, #0c1440);
    box-shadow: 0 0 25px #ffaa00; padding: 25px; border-radius: 20px;
}
button {
    background-color: #0c1440; color: #ffaa00; border: 2px solid #ffaa00; border-radius: 8px;
}
button:hover { background-color: #ffaa00; color: black; }
.chatbox {
    background-color: #111; color: #00ffff; border: 1px solid #00ffff; padding: 10px; height: 450px;
}
"""

with gr.Blocks(css=css) as demo:
    gr.Markdown("# 🧬 RILEY-AI")
    gr.Markdown("### Voice Enabled")

    chatbot = gr.Chatbot(label="Riley Terminal", elem_classes="chatbox", type='messages')
    msg = gr.Textbox(label="Ask or command Riley...")
    audio = gr.Audio(label="Riley’s Voice", interactive=False)
    clear = gr.Button("Clear Chat")
    state = gr.State([])

    msg.submit(chat_interface, [state, msg], [chatbot, msg, audio, state])
    clear.click(lambda: ([], "", None, []), None, [chatbot, msg, audio, state])

if __name__ == "__main__":
    demo.launch()
import gradio as gr
from models import ask_riley
from riley_genesis import RileyCore
import tempfile
from TTS.api import TTS

riley = RileyCore()
tts = TTS(model_name="tts_models/en/ljspeech/tacotron2-DDC", progress_bar=False)

def chat_interface(history, user_input):
    if user_input.startswith("!mode"):
        _, mode = user_input.split()
        return history + [{"role": "system", "content": riley.set_mode(mode)}], "", None, history

    if user_input.startswith("!personality"):
        _, profile = user_input.split()
        return history + [{"role": "system", "content": riley.set_personality(profile)}], "", None, history

    context_prompt = riley.think(user_input)
    response_raw = ask_riley(context_prompt)
    response = response_raw.replace('\n', ' ').replace('\r', '').replace('\\', '').strip()

    if "User:" in response:
        response = response.split("User:")[0].strip()

    riley.remember(f"Riley: {response}")
    history.append({"role": "user", "content": user_input})
    history.append({"role": "assistant", "content": response})

    audio_path = tempfile.mktemp(suffix=".wav")
    tts.tts_to_file(text=response, file_path=audio_path)

    return history, "", audio_path, history

hud_css = '''
body {
    background: radial-gradient(ellipse at center, #0f2027 0%, #203a43 50%, #2c5364 100%);
    font-family: 'Orbitron', sans-serif;
    color: #00f7ff;
}
.gradio-container {
    border: 2px solid #00f7ff;
    border-radius: 15px;
    background-color: #0c1440;
    box-shadow: 0 0 30px #00f7ff;
    padding: 30px;
}
button {
    background-color: #111;
    color: #00f7ff;
    border: 1px solid #00f7ff;
    padding: 10px 20px;
    border-radius: 8px;
}
button:hover {
    background-color: #00f7ff;
    color: #111;
}
.chatbox {
    background-color: #111;
    border: 1px solid #00f7ff;
    padding: 15px;
    color: #00f7ff;
    height: 500px;
}
'''

with gr.Blocks(css=hud_css) as demo:
    gr.Markdown("# 🌌 Riley AI - HUD Terminal")
    gr.Markdown("#### Type a message or command (e.g., `!mode logic`, `!personality commander`)")

    chatbot = gr.Chatbot(label="🧠 Riley Chat", elem_classes="chatbox", type='messages')
    msg = gr.Textbox(label="Command / Ask Riley...")
    audio = gr.Audio(label="Riley's Voice Output", interactive=False)
    clear = gr.Button("🧹 Clear")

    state = gr.State([])

    msg.submit(chat_interface, [state, msg], [chatbot, msg, audio, state])
    clear.click(lambda: ([], "", None, []), None, [chatbot, msg, audio, state])

if __name__ == "__main__":
    demo.launch(server_name="0.0.0.0", server_port=7860)
import gradio as gr
from TTS.api import TTS
import nltk
import tempfile

# Setup NLTK
nltk.download("punkt")
nltk.download("wordnet")

# Dummy RileyCore for this fix (replace with your real RileyCore later)
class RileyCore:
    def __init__(self):
        self.memory = []
        self.mode = "default"
        self.personality = "neutral"

    def think(self, prompt):
        return f"Mode: {self.mode} | Personality: {self.personality} | Prompt: {prompt}"

    def remember(self, thought):
        self.memory.append(thought)
        if len(self.memory) > 50:
            self.memory.pop(0)

    def set_mode(self, mode):
        self.mode = mode
        return f"Mode set to {mode}"

    def set_personality(self, profile):
        self.personality = profile
        return f"Personality set to {profile}"

riley = RileyCore()
tts = TTS(model_name="tts_models/en/ljspeech/tacotron2-DDC", progress_bar=False)

def respond(user_input, system_prompt, max_tokens, temperature, top_p):
    if not user_input.strip():
        return "Please enter a message.", None

    if user_input.startswith("!mode"):
        _, mode = user_input.split()
        return riley.set_mode(mode), None

    if user_input.startswith("!personality"):
        _, personality = user_input.split()
        return riley.set_personality(personality), None

    prompt = riley.think(user_input)
    response = f"Riley: {prompt}"
    riley.remember(response)

    audio_path = tempfile.mktemp(suffix=".wav")
    tts.tts_to_file(text=response, file_path=audio_path)

    return response, audio_path

demo = gr.ChatInterface(
    fn=respond,
    additional_inputs=[
        gr.Textbox(value="You are Riley.", label="System Prompt"),
        gr.Slider(1, 2048, value=512, label="Max tokens"),
        gr.Slider(0.1, 2.0, value=0.7, step=0.1, label="Temperature"),
        gr.Slider(0.1, 1.0, value=0.95, step=0.05, label="Top-p")
    ],
    title="🧠 Riley AI Interface",
    description="Riley is now active. Use `!mode scientist` or `!personality dark` to customize."
)

if __name__ == "__main__":
    demo.launch()

# Optimizations for resource-constrained environments
# 1. Use smaller or quantized models
from TTS.api import TTS

tts = TTS(model_name="tts_models/en/ljspeech/tacotron2-DDC", progress_bar=False, quantized=True)

# 2. Lazy load models to reduce memory usage during initialization
def get_riley_core():
    from riley_genesis import RileyCore
    return RileyCore()

riley = None

def initialize_riley():
    global riley
    if riley is None:
        riley = get_riley_core()

# 3. Reduce maximum token generation
MAX_TOKENS = 512

# 4. Limit TTS usage for constrained environments
def respond(user_input, system_prompt, max_tokens=MAX_TOKENS, temperature=0.7, top_p=0.95):
    initialize_riley()
    if not user_input.strip():
        return "Please enter a message.", None

    if user_input.startswith("!mode"):
        _, mode = user_input.split()
        return riley.set_mode(mode), None

    if user_input.startswith("!personality"):
        _, personality = user_input.split()
        return riley.set_personality(personality), None

    prompt = riley.think(user_input)
    response = f"Riley: {prompt}"
    riley.remember(response)

    # Only generate audio if explicitly requested
    audio_path = None
    if "!audio" in user_input:
        audio_path = tempfile.mktemp(suffix=".wav")
        tts.tts_to_file(text=response, file_path=audio_path)

    return response, audio_path
