import logging
import os
from typing import Any, List, Tuple

import gradio as gr
import torch
from dotenv import load_dotenv
from transformers import pipeline

from chat import Chat
from common import configs

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

load_dotenv()

logger.info(f"""Initializing model '{configs["model_id"]}'""")

pipe = pipeline(
    "text-generation",
    configs["model_id"],
    device=configs["device"],
    torch_dtype=torch.float16,
    token=os.environ["HF_ACCESS_TOKEN"],
)

chatbot_list: List[Chat] = [
    Chat(
        name="Seneca",
        init_prompt_path=configs["prompts"]["seneca"],
        model=pipe,
        max_new_tokens=configs["max_new_tokens"],
    ),
    Chat(
        name="Confucius",
        init_prompt_path=configs["prompts"]["confucius"],
        model=pipe,
        max_new_tokens=configs["max_new_tokens"],
    ),
    Chat(
        name="Aristotle",
        init_prompt_path=configs["prompts"]["aristotle"],
        model=pipe,
        max_new_tokens=configs["max_new_tokens"],
    ),
]


def chat_fn(prompt: str, _: List[str]) -> Tuple[str, list[Tuple[str, Any]]]:
    chat_history = []

    # Generate response from each bot
    for chatbot in chatbot_list:
        chatbot.continue_chat(prompt)
        chat_history.append(chatbot.chat_dict2list()[1:])

    merged_chat_history = []

    # Format chat from each bot
    for chat in zip(*chat_history):
        chat_history_ = []
        for chat_idx, (user_chat, assistant_chat) in enumerate(chat):
            if chat_idx == 0:
                chat_history_.append(
                    (user_chat, f"{chatbot_list[chat_idx].name}: {assistant_chat}")
                )
            else:
                chat_history_.append(
                    (None, f"{chatbot_list[chat_idx].name}: {assistant_chat}")
                )

        merged_chat_history.extend(chat_history_)

    logger.info("Chat updated")

    return "", merged_chat_history


with gr.Blocks() as demo:
    chatbot = gr.Chatbot(height=900)
    msg = gr.Textbox()
    clear = gr.ClearButton([msg, chatbot])
    msg.submit(chat_fn, [msg, chatbot], [msg, chatbot])

if __name__ == "__main__":
    demo.launch(
        server_port=int(os.environ["SERVER_PORT"]),
        server_name=os.environ["SERVER_NAME"],
    )
