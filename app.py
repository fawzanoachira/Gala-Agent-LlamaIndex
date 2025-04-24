from llama_index.core.agent.workflow import AgentWorkflow
from llama_index.llms.huggingface_api import HuggingFaceInferenceAPI
from llama_index.llms.fireworks import Fireworks
from tools import search_tool, weather_info_tool, hub_stats_tool
from retriever import guest_info_tool
from dotenv import load_dotenv
from huggingface_hub import InferenceClient

import gradio as gr
import asyncio, os

load_dotenv()
fw_api = os.getenv("fw_api")

# client = InferenceClient(
# 	provider="fireworks-ai",
#     model='Qwen/Qwen2.5-Coder-32B-Instruct',
# 	api_key=fw_api
# )

# model = HfApiModel(
#     model_id='Qwen/Qwen2.5-Coder-32B-Instruct',
#     provider='fireworks-ai',
#     token=fw_api,
# )

# llm = HuggingFaceInferenceAPI(model_name="Qwen/Qwen2.5-Coder-32B-Instruct")
llm = Fireworks(
    model="accounts/fireworks/models/deepseek-r1", api_key=fw_api
)

alfred = AgentWorkflow.from_tools_or_functions(
    [guest_info_tool, search_tool, weather_info_tool, hub_stats_tool],
    llm=llm,
)


async def answering(message, history):
    query = ""
    for msg in message:
        print(message)
        # if msg["role"] == "user" and isinstance(msg["content"], tuple):
            # query = msg["content"][0]
            # print(query)
    response = await alfred.run(message)
    return response


chat_ui = gr.ChatInterface(
    fn=answering,
    type="messages",
    title="Gala Agent LlamaIndex",
    textbox=gr.Textbox(),
)

if __name__ == "__main__":
    chat_ui.launch(debug=True)
