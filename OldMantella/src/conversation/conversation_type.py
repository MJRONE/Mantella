from abc import ABC, abstractmethod
import logging
from src.config.config_loader import ConfigLoader
from src.character_manager import Character
from src.llm.message_thread import message_thread
from src.conversation.context import Context
from src.llm.messages import UserMessage
from src import utils

class conversation_type(ABC):
    """Base class for different forms of conversations.
    """
    def __init__(self, config: ConfigLoader) -> None:
        super().__init__()
        self._config = config
    
    @abstractmethod
    def generate_prompt(self, context_for_conversation: Context) -> str:
        """Generates the text for the initial system_message. 

        Args:
            context_for_conversation (context): The context for the conversations. Provides tools to construct the prompt

        Returns:
            str: the prompt as a text
        """
        pass

    @abstractmethod
    def adjust_existing_message_thread(self, prompt: str, message_thread_to_adjust: message_thread):
        """Adjusts a given message_thread to one needed for the conversation type

        Args:
            prompt (str): The new prompt to add
            message_thread_to_adjust (message_thread): The message_thread to adjust
        """
        pass
    
    def get_user_message(self, context_for_conversation: Context, messages: message_thread) -> UserMessage | None:
        """Gets the next user message for the conversation. Default implementation gets the input from the player

        Args:
            settings (context): the current context of the conversation
            stt (Transcriber): the transcriber to get the voice input
            messages (message_thread): the current messages of the conversation

        Returns:
            user_message: the text for the next user message
        """
        return None
    
    def should_end(self, context_for_conversation: Context, messages: message_thread) -> bool:
        """Called after a message has been generated. Allows the conversation_type to stop the conversation at any point

        Args:
            settings (context): the current context of the conversation
            messages (message_thread): the current messages of the conversation
            game_state (GameStateManager): the GameStateManager to make inquiries or send messages to Skyrim if needed

        Returns:
            bool: True if the conversation should end, False otherwise
        """
        return False

class pc_to_npc(conversation_type):
    """PC talks to a single NPC. The classic conversation"""
    def __init__(self, config: ConfigLoader) -> None:
        super().__init__(config)

    @utils.time_it
    def generate_prompt(self, context_for_conversation: Context) -> str:
        actions = [a for a in self._config.actions if a.use_in_on_on_one]
        return context_for_conversation.generate_system_message(self._config.prompt, actions)
    
    @utils.time_it
    def adjust_existing_message_thread(self, prompt: str, message_thread_to_adjust: message_thread):
        message_thread_to_adjust.modify_messages(prompt, multi_npc_conversation=False, remove_system_flagged_messages=True)
    
    @utils.time_it
    def get_user_message(self, context_for_conversation: Context, messages: message_thread) -> UserMessage | None:
        if len(messages) == 1 and context_for_conversation.config.automatic_greeting:
            player_character: Character | None = context_for_conversation.npcs_in_conversation.get_player_character()
            if player_character:
                for actor in context_for_conversation.npcs_in_conversation.get_all_characters():
                    if not actor.is_player_character:
                        message = UserMessage(context_for_conversation.config, f"{context_for_conversation.language['hello']} {actor.name}.", player_character.name, True)
                        message.is_multi_npc_message = False
                        return message
            return None
        else:
            return super().get_user_message(context_for_conversation, messages)

class multi_npc(conversation_type):
    """Group conversation between the PC and multiple NPCs"""
    def __init__(self, config: ConfigLoader) -> None:
        super().__init__(config)

    @utils.time_it
    def generate_prompt(self, context_for_conversation: Context) -> str:
        actions = [a for a in self._config.actions if a.use_in_multi_npc]
        return context_for_conversation.generate_system_message(self._config.multi_npc_prompt, actions)
    
    @utils.time_it
    def adjust_existing_message_thread(self, prompt: str, message_thread_to_adjust: message_thread):
        message_thread_to_adjust.modify_messages(prompt, True, True)

    @utils.time_it
    def get_user_message(self, context_for_conversation: Context, messages: message_thread) -> UserMessage | None:
        if len(messages) == 1 and context_for_conversation.config.automatic_greeting:
            player_character: Character | None = context_for_conversation.npcs_in_conversation.get_player_character()
            if player_character:
                message = UserMessage(context_for_conversation.config, f"{context_for_conversation.language['hello']} {context_for_conversation.get_character_names_as_text(should_include_player=False)}.", player_character.name, True)
                message.is_multi_npc_message = True
                return message
            return None
        else:
            return super().get_user_message(context_for_conversation, messages)

class radiant(conversation_type):
    """ Conversation between two NPCs without the player"""
    def __init__(self, config: ConfigLoader) -> None:
        super().__init__(config)
        self.__user_start_prompt = config.radiant_start_prompt
        self.__user_end_prompt = config.radiant_end_prompt
        self.__has_started = False
        self.__sent_end_prompt = False

    @utils.time_it
    def generate_prompt(self, context_for_conversation: Context) -> str:
        actions = [a for a in self._config.actions if a.use_in_radiant]
        # Fallback: if no actions explicitly enabled for radiant, include multi_npc and one_on_one actions
        if len(actions) == 0:
            try:
                actions = [a for a in self._config.actions if a.use_in_multi_npc or a.use_in_on_on_one]
            except Exception:
                pass
        try:
            import logging
            logging.log(23, f"Radiant actions included in prompt: {len(actions)}")
        except Exception:
            pass
        return context_for_conversation.generate_system_message(self._config.radiant_prompt, actions)
    
    @utils.time_it
    def adjust_existing_message_thread(self, prompt: str, message_thread_to_adjust: message_thread):
        message_thread_to_adjust.modify_messages(prompt, True, True)
    
    def request_end_prompt(self) -> None:
        """Called when conversation should start wrapping up"""
        self.__sent_end_prompt = True
        logging.info(f"[RADIANT] End prompt requested")
    
    def has_sent_end_prompt(self) -> bool:
        """Check if end prompt has been sent"""
        return self.__sent_end_prompt
    
    @utils.time_it
    def get_user_message(self, context_for_conversation: Context, messages: message_thread) -> UserMessage | None:        
        # Check if we need to send the end prompt
        if self.__sent_end_prompt:
            # Check if end prompt has already been sent by looking at messages
            user_messages = [msg for msg in messages.get_talk_only(include_system_generated_messages=True) if isinstance(msg, UserMessage)]
            if user_messages:
                last_user_msg = user_messages[-1]
                if self.__user_end_prompt in last_user_msg.text:
                    # End prompt already sent, return None to end conversation
                    return None
            
            # Send the end prompt
            reply = UserMessage(context_for_conversation.config, self.__user_end_prompt, "", False)
            reply.is_multi_npc_message = False
            logging.info(f"[RADIANT] Sending end prompt (time limit reached)")
            return reply
        
        # For the first prompt, send the start prompt
        if not self.__has_started:
            self.__has_started = True
            reply = UserMessage(context_for_conversation.config, self.__user_start_prompt, "", False)
            reply.is_multi_npc_message = False
            logging.info(f"[RADIANT] Sending start prompt")
            return reply
        
        # For subsequent prompts, send continue prompt
        reply = UserMessage(context_for_conversation.config, "Continue the conversation naturally.", "", False)
        reply.is_multi_npc_message = False
        logging.info(f"[RADIANT] Sending continue prompt")
        return reply
    
    def should_end(self, context_for_conversation: Context, messages: message_thread) -> bool:
        # Radiant conversations need at least 2 NPCs (no player)
        # End immediately if not enough participants remain
        npc_count = context_for_conversation.npcs_in_conversation.active_character_count()
        if npc_count < 2:
            logging.info(f"[RADIANT] Ending conversation - insufficient participants ({npc_count} < 2)")
            return True
        
        # End if we've sent the end prompt and got a response
        if self.__sent_end_prompt:
            logging.info(f"[RADIANT] Ending conversation - end prompt completed")
            return True
        
        return False