from llm.base import LLMProvider, Message
from logger import get_logger

logger = get_logger(__name__)

class Assistant:
    
    def __init__(
        self,
        llm: LLMProvider,
        system_prompt: str
    ):
        self.llm = llm
        self.system_prompt = system_prompt
        
        self.messages: list[Message] = [
            {
                "role": "system",
                "content": self.system_prompt
            }
        ]
    
    def send_message(self, user_message: str, model_message: str | str | None = none):

        user_entry = {
                "role": "user",
                "content": user_message
        }
        
        self.messages.append(user_entry)
        
        message_for_model = (
            model_message
            if model_message is not None
            else user_message
        )

        request_messages = [
            *self.messages[:-1],
            {
                "role": "user",
                "content": message_for_model
            }
        ]

        response= ""
        
        try:
            for text in self.llm.stream_chat(self.messages):
                response += text
                yield text

        except Exception:
            self.messages.pop()
            raise

        self.messages.append(
            {
                "role": "assistant",
                "content": response
            }
        )
    
    def change_model(self, model: str) -> None:
        self.llm.set_model(model)
        self.reset()
    
    def reset(self):
        
        self.messages = [
            {
                "role": "system",
                "content": self.system_prompt
            }
        ]

        logger.info("Conversation reset")
    
    