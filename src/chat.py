import logging
from typing import Any, Dict, List

from pydantic import BaseModel


class Chat(BaseModel):
    name: str
    init_prompt_path: str
    chat: List[Dict] = []
    model: Any
    max_new_tokens: int

    def model_post_init(self, __context: Any) -> None:
        logger.info(f"Initializing chatbot '{self.name}'")
        init_prompt = open(self.init_prompt_path, "r").read()
        self.continue_chat(init_prompt)

    def continue_chat(self, prompt: str) -> None:
        logger.info(f"Chatbot '{self.name}' processing prompt '{prompt}'")
        self.chat.append(
            {
                "role": "user",
                "content": prompt,
            }
        )
        self.chat = self.model(
            self.chat,
            max_new_tokens=self.max_new_tokens,
        )[
            0
        ]["generated_text"]

    def chat_dict2list(self) -> List[str]:
        chat_list = [x["content"] for x in self.chat]
        chat_list = list(zip(chat_list[::2], chat_list[1::2]))
        return chat_list


logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)
