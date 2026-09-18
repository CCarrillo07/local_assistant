from llm.base import LLMProvider, Message
from logger import get_logger
from memory.base import ConversationMemory
from memory.short_term import SlidingWindowMemory

logger = get_logger(__name__)

class Assistant:
    
    def __init__(
        self,
        llm: LLMProvider,
        system_prompt: str,
        memory: ConversationMemory | None = None
    ):
        self.llm = llm
        self.system_prompt = system_prompt

        self.memory = (
            memory
            if memory is not None
            else SlidingWindowMemory()
        )
        
    def send_message(
            self, 
            user_message: str, 
            model_message: str | None = None,
            context_messages: list[Message] | None = None
    ):

        message_for_model = (
            model_message
            if model_message is not None
            else user_message
        )

        additional_context = (
            context_messages
            if context_messages is not None
            else []  
        )
        
        request_messages: list[Message] = [
            {
                "role": "system",
                "content": self.system_prompt
            },
            *additional_context,
            *self.memory.get_messages(),
            {
                "role": "user",
                "content": message_for_model
            }
        ]

        response = ""

        for text in self.llm.stream_chat(request_messages):
            response += text
            yield text

        self.memory.add_turn(
            user_message=user_message,
            assistant_message=response
        )
    
    def change_model(self, model: str) -> None:
        self.llm.set_model(model)
        self.reset()
    
    def reset(self):
        self.memory.clear()
        logger.info("Conversation reset")