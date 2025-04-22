from llama_index.core.agent.workflow import AgentWorkflow
from llama_index.llms.huggingface_api import HuggingFaceInferenceAPI
from tools import search_tool, weather_info_tool, hub_stats_tool
from retriever import guest_info_tool

import gradio as gr

llm = HuggingFaceInferenceAPI(model_name="Qwen/Qwen2.5-Coder-32B-Instruct")

alfred = AgentWorkflow.from_tools_or_functions(
    [guest_info_tool, search_tool, weather_info_tool, hub_stats_tool],
    llm=llm,
)


def answering(message, history):
    query = ""
    for msg in history:
        if msg["role"] == "user" and isinstance(msg["content"], tuple):
            query = msg["content"][0]
    response = alfred.run(query)
    return response


chat_ui = gr.ChatInterface(
    answering,
    type="messages",
    title="Gala Agent LlamaIndex",
    textbox=gr.Textbox(),
)

if __name__ == "__main__":
    chat_ui.launch(debug=True)
