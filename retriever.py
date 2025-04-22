import datasets


from llama_index.core.agent.workflow import AgentWorkflow
from llama_index.llms.huggingface_api import HuggingFaceInferenceAPI
from llama_index.core.tools import FunctionTool
from llama_index.retrievers.bm25 import BM25Retriever
from llama_index.core.schema import Document

guest_dataset = datasets.load_dataset("agents-course/unit3-invitees", split="train")

docs = [
    Document(
        text="\n".join(
            [
                f"Name: {guest_dataset['name'][i]}",
                f"Relation: {guest_dataset['relation'][i]}",
                f"Description: {guest_dataset['description'][i]}",
                f"Email: {guest_dataset['email'][i]}",
            ]
        ),
        metadata={"name": guest_dataset["name"][i]},
    )
    for i in range(len(guest_dataset))
]

bm25_retriever = BM25Retriever.from_defaults(nodes=docs)


def get_guest_info_retriever(query: str) -> str:
    """Retrieves detailed information about gala guests based on their name or relation."""
    results = bm25_retriever.retrieve(query)
    if results:
        return "\n\n".join([doc.text for doc in results[:3]])
    else:
        return "No matching guest information found."


guest_info_tool = FunctionTool.from_defaults(get_guest_info_retriever)
