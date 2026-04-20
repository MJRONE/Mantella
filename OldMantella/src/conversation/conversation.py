from enum import Enum
import logging
from threading import Thread, Lock
import time
from typing import Any
from src.llm.ai_client import AIClient
from src.llm.sentence_content import SentenceTypeEnum, SentenceContent
from src.characters_manager import Characters
from src.conversation.conversation_log import conversation_log
from src.conversation.action import Action
from src.llm.sentence_queue import SentenceQueue
from src.llm.sentence import Sentence
from src.remember.remembering import Remembering
from src.output_manager import ChatManager
from src.llm.messages import AssistantMessage, SystemMessage, UserMessage
from src.conversation.context import Context
from src.llm.message_thread import message_thread
from src.conversation.conversation_type import conversation_type, multi_npc, pc_to_npc, radiant
from src.character_manager import Character
from src.http.communication_constants import communication_constants as comm_consts
from src.stt import Transcriber
import src.utils as utils

class conversation_continue_type(Enum):
    NPC_TALK = 1
    PLAYER_TALK = 2
    END_CONVERSATION = 3

class Conversation:
    TOKEN_LIMIT_PERCENT: float = 0.9
    TOKEN_LIMIT_RELOAD_MESSAGES: float = 0.1
    """Controls the flow of a conversation."""
    def __init__(self, context_for_conversation: Context, output_manager: ChatManager, rememberer: Remembering, llm_client: AIClient, stt: Transcriber | None, mic_input: bool, mic_ptt: bool) -> None:
        
        self.__context: Context = context_for_conversation
        self.__mic_input: bool = mic_input
        self.__mic_ptt: bool = mic_ptt
        self.__allow_interruption: bool = context_for_conversation.config.allow_interruption # allow mic interruption
        self.__is_player_interrupting = False
        self.__stt: Transcriber | None = stt
        self.__events_refresh_time: float = context_for_conversation.config.events_refresh_time  # Time in seconds before events are considered stale
        self.__transcribed_text: str | None = None
        if not self.__context.npcs_in_conversation.contains_player_character(): # TODO: fix this being set to a radiant conversation because of NPCs in conversation not yet being added
            self.__conversation_type: conversation_type = radiant(context_for_conversation.config)
            # Reset processed events count for new radiant conversation
            self.__processed_radiant_events_count = 0
            # Track start time for duration-based radiant conversations
            self.__radiant_start_time = time.time()
            self.__radiant_end_requested = False
        else:
            self.__conversation_type: conversation_type = pc_to_npc(context_for_conversation.config)
            self.__radiant_start_time = 0
            self.__radiant_end_requested = False        
        self.__messages: message_thread = message_thread(self.__context.config, None)
        self.__output_manager: ChatManager = output_manager
        self.__rememberer: Remembering = rememberer
        self.__llm_client = llm_client
        self.__has_already_ended: bool = False
        self.__allow_mic_input: bool = True # this flag ensures mic input is disabled on conversation end
        self.__sentences: SentenceQueue = SentenceQueue()
        self.__generation_thread: Thread | None = None
        self.__generation_start_lock: Lock = Lock()
        # self.__actions: list[Action] = actions
        self.last_sentence_audio_length = 0
        self.last_sentence_start_time = time.time()
        self.__end_conversation_keywords = utils.parse_keywords(context_for_conversation.config.end_conversation_keyword)
        self.__processed_radiant_events_count: int = 0  # Track how many events have been processed in current conversation
        self.__vision_requested_for_current_conversation: bool = False  # Track if vision was requested for current conversation
        self.__last_vision_trigger_time: float = 0  # Track when vision was last triggered for periodic updates
        self.__last_vision_used_time: float = 0  # Track when the last vision result was used in an LLM prompt
        
        # Debounce mechanism to prevent duplicate LLM calls from simultaneous events/vision
        self.__pending_injection_lock: Lock = Lock()
        self.__pending_injection_events: list[str] = []
        self.__pending_injection_timer: Thread | None = None
        self.__injection_debounce_seconds: float = 0.3  # Wait 300ms to batch simultaneous injections
        self.__allow_vision_injection: bool = False  # Prevents vision injection during initial startup

    @property
    def has_already_ended(self) -> bool:
        return self.__has_already_ended

    def inject_late_vision_description(self, vision_description: str) -> bool:
        """Inject a late-arriving vision description into an ongoing radiant conversation

        Args:
            vision_description: The vision description to inject

        Returns:
            bool: True if successfully injected, False if injection failed
        """
        if self.__has_already_ended:
            logging.warning("[RADIANT VISION] LATE - Cannot inject vision: conversation has already ended")
            return False

        if not isinstance(self.__conversation_type, radiant):
            logging.warning("[RADIANT VISION] LATE - Cannot inject vision: not a radiant conversation")
            return False
        
        # IMPORTANT: Block vision injection during initial startup to prevent duplicate LLM calls
        if not self.__allow_vision_injection:
            logging.debug("[RADIANT VISION] LATE - Blocking vision injection during initial startup phase")
            return False

        # Create vision event and batch it with any other pending injections
        vision_event = f"Late visual context as seen by MJR (the player): {vision_description}"
        
        logging.info(f"[RADIANT VISION] LATE - Queueing late vision for batched injection")
        
        # Mark this vision as used since it's being queued for injection
        self.__last_vision_used_time = time.time()

        # Use debounce mechanism to batch with other simultaneous injections
        self.__queue_injection([vision_event])
        return True

    def __trigger_deferred_initial_vision(self):
        """Trigger initial vision capture for radiant conversations AFTER startup (non-blocking)"""
        def capture_vision():
            try:
                if (hasattr(self.__llm_client, '_image_client') and 
                    self.__llm_client._image_client and 
                    not self.__has_already_ended):
                    
                    logging.debug("[RADIANT VISION] DEFERRED - Starting non-blocking vision capture after startup")
                    
                    # IMPORTANT: Wait to ensure the initial greeting LLM call completes first
                    # This prevents the vision from arriving during the initial response
                    time.sleep(3.0)
                    
                    if self.__has_already_ended:
                        return
                    
                    # Enable vision injection now that initial greeting is done
                    self.__allow_vision_injection = True
                    logging.debug("[RADIANT VISION] DEFERRED - Vision injection enabled after initial startup")
                    
                    # Capture vision without timeout (async via callback)
                    vision_description = self.__llm_client._image_client.get_vision_description_with_timeout(
                        vision_hints="Describe what you see in the current scene as seen by MJR (the player) for NPCs to discuss.",
                        timeout_seconds=0  # Non-blocking
                    )
                    
                    if vision_description and not self.__has_already_ended:
                        # Got vision, inject it via the callback mechanism
                        logging.debug("[RADIANT VISION] DEFERRED - Vision obtained, injecting")
                        self.inject_late_vision_description(vision_description)
                    else:
                        # Vision will arrive later via callback, or not at all
                        logging.debug("[RADIANT VISION] DEFERRED - Vision will arrive via callback (if available)")
            except Exception as e:
                logging.error(f"[RADIANT VISION] DEFERRED - Error capturing deferred vision: {e}")
        
        # Run vision capture in background thread to not block conversation startup
        vision_thread = Thread(target=capture_vision, daemon=True)
        vision_thread.start()
    
    @utils.time_it
    def __trigger_periodic_vision(self) -> str | None:
        """Trigger vision during radiant conversations for periodic updates

        Returns:
            str | None: Vision description if successful, None otherwise
        """
        if not self.__context.config.vision_enabled:
            return None

        if not hasattr(self.__llm_client, '_image_client') or not self.__llm_client._image_client:
            return None

        try:
            # Temporarily clear the late vision callback to avoid conflicts with periodic vision
            original_callback = None
            if hasattr(self.__llm_client._image_client, '_late_vision_callback'):
                original_callback = self.__llm_client._image_client._late_vision_callback
                self.__llm_client._image_client.set_late_vision_callback(None)

            try:
                # Get vision description without timeout for periodic updates to avoid delaying conversation flow
                vision_description = self.__llm_client._image_client.get_vision_description_with_timeout(
                    vision_hints="Describe what you see in the current scene as seen by MJR (the player) for NPCs to discuss.",
                    timeout_seconds=0  # No timeout for periodic updates
                )

                if vision_description:
                    return vision_description
                else:
                    logging.debug("[RADIANT VISION] PERIODIC - No vision description obtained for periodic update")

            finally:
                # Restore the original late vision callback
                if original_callback:
                    self.__llm_client._image_client.set_late_vision_callback(original_callback)

        except Exception as e:
            logging.error(f"[RADIANT VISION] PERIODIC - Failed to get periodic vision description: {e}")
            import traceback
            logging.error(f"[RADIANT VISION] PERIODIC - Full traceback: {traceback.format_exc()}")

        return None

    def get_radiant_conversation_debug_info(self) -> str:
        """Get debug information about radiant conversation state"""
        is_radiant = isinstance(self.__conversation_type, radiant)
        radiant_events = self.__context.get_radiant_event_log()
        processed_count = self.__processed_radiant_events_count
        new_events_count = len(radiant_events) - processed_count

        return (f"Radiant Conversation Debug:\n"
                f"  Is Radiant: {is_radiant}\n"
                f"  Total Events in Log: {len(radiant_events)}\n"
                f"  Processed Events: {processed_count}\n"
                f"  New Events Available: {new_events_count}\n"
                f"  Conversation Messages: {len(self.__messages)}\n"
                f"  Vision Enabled: {self.__context.config.vision_enabled}\n"
                f"  Vision Requested: {self.__vision_requested_for_current_conversation}")
    
    @property
    def context(self) -> Context:
        return self.__context
    
    @property
    def output_manager(self) -> ChatManager:
        return self.__output_manager
    
    @property
    def transcribed_text(self) -> str | None:
        return self.__transcribed_text
    
    @property
    def stt(self) -> Transcriber | None:
        return self.__stt
    
    @utils.time_it
    def add_or_update_character(self, new_character: list[Character]):
        """Adds or updates a character in the conversation.

        Args:
            new_character (Character): the character to add or update
        """
        characters_removed_by_update = self.__context.add_or_update_characters(new_character)
        if len(characters_removed_by_update) > 0:
            all_characters = self.__context.npcs_in_conversation.get_all_characters()
            all_characters.extend(characters_removed_by_update)
            self.__save_conversations_for_characters(all_characters, is_reload=True)
            
            # For radiant conversations, check if we still have enough NPCs after removal
            if isinstance(self.__conversation_type, radiant):
                npc_count = self.__context.npcs_in_conversation.active_character_count()
                if npc_count < 2:
                    logging.info(f"[RADIANT] NPCs removed via update - only {npc_count} NPCs left, ending conversation")
                    self.initiate_end_sequence()

    @utils.time_it
    def start_conversation(self) -> tuple[str, Sentence | None]:
        """Starts a new conversation.

        Returns:
            tuple[str, sentence | None]: Returns a tuple consisting of a reply type and an optional sentence
        """
        greeting: UserMessage | None = self.__conversation_type.get_user_message(self.__context, self.__messages)
        if greeting:
            self.__messages.add_message(greeting)
            
            # For radiant conversations, trigger vision capture AFTER startup (non-blocking)
            if isinstance(self.__conversation_type, radiant) and self.__vision_requested_for_current_conversation:
                self.__trigger_deferred_initial_vision()
            
            self.__start_generating_npc_sentences()
            return comm_consts.KEY_REPLYTYPE_NPCTALK, None
        else:
            return comm_consts.KEY_REPLYTYPE_PLAYERTALK, None

    @utils.time_it
    def continue_conversation(self) -> tuple[str, Sentence | None]:
        """Main workhorse of the conversation. Decides what happens next based on the state of the conversation

        Returns:
            tuple[str, sentence | None]: Returns a tuple consisting of a reply type and an optional sentence
        """
        if self.has_already_ended:
            return comm_consts.KEY_REPLYTYPE_ENDCONVERSATION, None
        
        # For radiant conversations, check if we still have enough NPCs
        if isinstance(self.__conversation_type, radiant):
            npc_count = self.__context.npcs_in_conversation.active_character_count()
            if npc_count < 2:
                logging.info(f"[RADIANT] Not enough NPCs to continue ({npc_count} < 2), ending conversation")
                self.initiate_end_sequence()
                return comm_consts.KEY_REPLYTYPE_ENDCONVERSATION, None
        
        if self.__llm_client.is_too_long(self.__messages, self.TOKEN_LIMIT_PERCENT):
            # Check if conversation too long and if yes initiate intermittent reload
            self.__initiate_reload_conversation()

        # interrupt response if player has spoken
        if self.__stt and self.__stt.has_player_spoken:
            self.__stop_generation()
            self.__sentences.clear()
            self.__is_player_interrupting = True
            return comm_consts.KEY_REQUESTTYPE_TTS, None
        
        # restart mic listening as soon as NPC's first sentence is processed
        if self.__mic_input and self.__allow_interruption and not self.__mic_ptt and not self.__stt.is_listening and self.__allow_mic_input and not isinstance(self.__conversation_type, radiant):
            mic_prompt = self.__get_mic_prompt()
            self.__stt.start_listening(mic_prompt)
        
        #Grab the next sentence from the queue
        next_sentence: Sentence | None = self.retrieve_sentence_from_queue()
        
        if next_sentence and len(next_sentence.text) > 0:
            # Events for radiant conversations are now handled at conversation start only
            if comm_consts.ACTION_REMOVECHARACTER in next_sentence.actions:
                self.__context.remove_character(next_sentence.speaker)
                # For radiant conversations, check if we still have enough NPCs
                if isinstance(self.__conversation_type, radiant):
                    npc_count = self.__context.npcs_in_conversation.active_character_count()
                    if npc_count < 2:
                        logging.info(f"[RADIANT] NPC removed - only {npc_count} NPCs left, ending conversation immediately")
                        self.initiate_end_sequence()
                        return comm_consts.KEY_REPLYTYPE_ENDCONVERSATION, None
            #if there is a next sentence and it actually has content, return it as something for an NPC to say
            if self.last_sentence_audio_length > 0:
                logging.debug(f'Waiting {round(self.last_sentence_audio_length, 1)} seconds for last voiceline to play')
            # before immediately sending the next voiceline, give the player the chance to interrupt
            while time.time() - self.last_sentence_start_time < self.last_sentence_audio_length:
                if self.__stt and self.__stt.has_player_spoken:
                    self.__stop_generation()
                    self.__sentences.clear()
                    self.__is_player_interrupting = True
                    return comm_consts.KEY_REQUESTTYPE_TTS, None
                # Longer sleep for radiant conversations since no player interaction
                sleep_duration = 0.05 if isinstance(self.__conversation_type, radiant) else 0.01
                time.sleep(sleep_duration)
            self.last_sentence_audio_length = next_sentence.voice_line_duration + self.__context.config.wait_time_buffer
            self.last_sentence_start_time = time.time()
            return comm_consts.KEY_REPLYTYPE_NPCTALK, next_sentence
        else:
            #Ask the conversation type here, if we should end the conversation
            if self.__conversation_type.should_end(self.__context, self.__messages):
                # Only end if queue is empty or nearly empty (to avoid cutting off pending sentences)
                if self.__sentences.get_size() <= 1:  # Allow for empty sentinel
                    self.initiate_end_sequence()
                    return comm_consts.KEY_REPLYTYPE_NPCTALK, None
                # Otherwise keep processing queue before ending
            
            # For radiant conversations, check if we should send end prompt due to time limit
            if isinstance(self.__conversation_type, radiant) and not self.__radiant_end_requested:
                if self.__context.config.max_radiant_duration > 0:
                    elapsed_time = time.time() - self.__radiant_start_time
                    if elapsed_time >= self.__context.config.max_radiant_duration:
                        logging.info(f"[RADIANT] Duration limit reached ({elapsed_time:.1f}s >= {self.__context.config.max_radiant_duration}s) - requesting end prompt")
                        self.__conversation_type.request_end_prompt()
                        self.__radiant_end_requested = True
            
            #If not ended, ask the conversation type for an automatic user message. If there is None, signal the game that the player must provide it 
            new_user_message = self.__conversation_type.get_user_message(self.__context, self.__messages)
            if new_user_message:
                self.__messages.add_message(new_user_message)
                self.__start_generating_npc_sentences()
                return comm_consts.KEY_REPLYTYPE_NPCTALK, None
            else:
                return comm_consts.KEY_REPLYTYPE_PLAYERTALK, None

    @utils.time_it
    def process_player_input(self, player_text: str) -> tuple[str, bool, Sentence|None]:
        """Submit the input of the player to the conversation

        Args:
            player_text (str): The input text / voice transcribe of what the player character is supposed to say. Can be empty if mic input has not yet been parsed

        Returns:
            tuple[str, bool]: Returns a tuple consisting of updated player text (if using mic input) and whether or not in-game events need to be refreshed (depending on how much time has passed)
        """
        player_character = self.__context.npcs_in_conversation.get_player_character()
        if not player_character:
            return '', False, None # If there is no player in the conversation, exit here
        
        events_need_updating: bool = False

        with self.__generation_start_lock: #This lock makes sure no new generation by the LLM is started while we clear this
            self.__stop_generation() # Stop generation of additional sentences right now
            self.__sentences.clear() # Clear any remaining sentences from the list

            # If the player's input does not already exist, parse mic input if mic is enabled
            if self.__mic_input and len(player_text) == 0:
                player_text = None
                if not self.__stt.is_listening and self.__allow_mic_input:
                    self.__stt.start_listening(self.__get_mic_prompt())
                
                # Start tracking how long it has taken to receive a player response
                input_wait_start_time = time.time()
                while not player_text:
                    player_text = self.__stt.get_latest_transcription()
                if time.time() - input_wait_start_time >= self.__events_refresh_time:
                    # If too much time has passed, in-game events need to be updated
                    events_need_updating = True
                    logging.debug('Updating game events...')
                    return player_text, events_need_updating, None
                
                # Stop listening once input has been detected to give the NPC a chance to speak
                # This also needs to apply when interruptions are allowed, 
                # otherwise the player could constantly speak over the NPC and never hear a response
                self.__stt.stop_listening()
            
            new_message: UserMessage = UserMessage(self.__context.config, player_text, player_character.name, False)
            new_message.is_multi_npc_message = self.__context.npcs_in_conversation.contains_multiple_npcs()
            new_message = self.update_game_events(new_message)
            self.__messages.add_message(new_message)
            player_voiceline = self.__get_player_voiceline(player_character, player_text)
            text = new_message.text
            logging.log(23, f"Text passed to NPC: {text}")

        ejected_npc = self.__does_dismiss_npc_from_conversation(text)
        if ejected_npc:
            self.__prepare_eject_npc_from_conversation(ejected_npc)
        elif self.__has_conversation_ended(text):
            new_message.is_system_generated_message = True # Flag message containing goodbye as a system message to exclude from summary
            self.initiate_end_sequence()
        else:
            self.__start_generating_npc_sentences()

        return player_text, events_need_updating, player_voiceline

    def __get_mic_prompt(self):
        mic_prompt = f"This is a conversation with {self.__context.get_character_names_as_text(False)} in {self.__context.location}."
        #logging.log(23, f'Context for mic transcription: {mic_prompt}')
        return mic_prompt
    
    @utils.time_it
    def __get_player_voiceline(self, player_character: Character | None, player_text: str) -> Sentence | None:
        """Synthesizes the player's input if player voice input is enabled, or else returns None
        """
        player_character_voiced_sentence: Sentence | None = None
        if self.__should_voice_player_input(player_character):
            player_character_voiced_sentence = self.__output_manager.generate_sentence(SentenceContent(player_character, player_text, SentenceTypeEnum.SPEECH, False))
            if player_character_voiced_sentence.error_message:
                player_message_content: SentenceContent = SentenceContent(player_character, player_text, SentenceTypeEnum.SPEECH, False)
                player_character_voiced_sentence = Sentence(player_message_content, "" , 2.0)

        return player_character_voiced_sentence

    @utils.time_it
    def update_context(self, location: str | None, time: int, custom_ingame_events: list[str], weather: str, custom_context_values: dict[str, Any]):
        """Updates the context with a new set of values

        Args:
            location (str): the location the characters are currently in
            time (int): the current ingame time
            custom_ingame_events (list[str]): a list of events that happend since the last update
            custom_context_values (dict[str, Any]): the current set of context values
        """
        # Removed debug logging for performance
        self.__context.update_context(location, time, custom_ingame_events, weather, custom_context_values)
        if self.__context.have_actors_changed:
            self.__update_conversation_type()
            self.__context.have_actors_changed = False

    @utils.time_it
    def __update_conversation_type(self):
        """This changes between pc_to_npc, multi_npc and radiant conversation_types based on the current state of the context
        """
        # If the conversation can proceed for the first time, it starts and we add the system_message with the prompt
        if not self.__has_already_ended:
            self.__stop_generation()
            self.__sentences.clear()
            
            if not self.__context.npcs_in_conversation.contains_player_character():
                self.__conversation_type = radiant(self.__context.config)
                # Reset processed events count for new radiant conversation
                self.__processed_radiant_events_count = 0
                # Clear vision requested flag for new conversation
                self.__vision_requested_for_current_conversation = False
                # Reset vision timers for new radiant conversation
                self.__last_vision_trigger_time = 0
                self.__last_vision_used_time = 0
                # Reset vision injection flag for new radiant conversation
                self.__allow_vision_injection = False
                # Clear any existing late vision callback
                if hasattr(self.__llm_client, '_image_client') and self.__llm_client._image_client:
                    self.__llm_client._image_client.set_late_vision_callback(None)
            elif self.__context.npcs_in_conversation.active_character_count() >= 3:
                self.__conversation_type = multi_npc(self.__context.config)
                # Clear vision callback when changing from radiant
                if hasattr(self.__llm_client, '_image_client') and self.__llm_client._image_client:
                    self.__llm_client._image_client.set_late_vision_callback(None)
                # Reset vision timers when changing from radiant
                self.__last_vision_trigger_time = 0
                self.__last_vision_used_time = 0
                self.__allow_vision_injection = False
            else:
                self.__conversation_type = pc_to_npc(self.__context.config)
                # Clear vision callback when changing from radiant
                if hasattr(self.__llm_client, '_image_client') and self.__llm_client._image_client:
                    self.__llm_client._image_client.set_late_vision_callback(None)
                # Reset vision timers when changing from radiant
                self.__last_vision_trigger_time = 0
                self.__last_vision_used_time = 0
                self.__allow_vision_injection = False

            new_prompt = self.__conversation_type.generate_prompt(self.__context)        
            if len(self.__messages) == 0:
                self.__messages: message_thread = message_thread(self.__context.config, new_prompt)
                # For radiant conversations, inject all accumulated events from the persistent log at start
                if isinstance(self.__conversation_type, radiant):
                    # Get all accumulated events from radiant log
                    all_radiant_events = self.__context.get_radiant_event_log()
                    logging.info(f"[RADIANT EVENTS] START - Retrieved {len(all_radiant_events)} events from radiant log at conversation start")
                    if len(all_radiant_events) > 0:
                        logging.info(f"[RADIANT EVENTS] START - Events to be used: {all_radiant_events}")

                    # Also check if there are any recent regular events that haven't been processed yet
                    recent_regular_events = self.__context.get_context_ingame_events()
                    if len(recent_regular_events) > 0:
                        logging.info(f"[RADIANT EVENTS] START - Found {len(recent_regular_events)} recent regular events: {recent_regular_events}")
                        # Add recent events to radiant log if they're not already there
                        for event in recent_regular_events:
                            if event not in all_radiant_events:
                                all_radiant_events.append(event)
                                logging.info(f"[RADIANT EVENTS] START - Added recent event to radiant log: {event}")

                    # OPTIMIZATION: Move vision capture OUT of startup critical path
                    # Set up vision callback but don't capture yet - will trigger after first message
                    if (self.__context.config.vision_enabled and
                        hasattr(self.__llm_client, '_image_client') and
                        self.__llm_client._image_client):
                        # Mark that vision was requested for this conversation
                        self.__vision_requested_for_current_conversation = True
                        # Set up callback for late vision injection
                        self.__llm_client._image_client.set_late_vision_callback(self.inject_late_vision_description)
                        logging.debug("[RADIANT VISION] STARTUP - Vision callback registered, will capture after first message")
                    
                    # Initialize timer for periodic vision
                    self.__last_vision_trigger_time = time.time()
                    
                    # Create event message with all accumulated events (vision will be added later via callback)
                    if len(all_radiant_events) > 0:
                        event_only_message = UserMessage(self.__context.config, "", "", False)
                        
                        # Apply max_count_events limit to initial events
                        original_event_count = len(all_radiant_events)
                        max_events = self.__context.config.max_count_events
                        
                        if max_events > 0 and len(all_radiant_events) > max_events:
                            # Keep only the most recent events
                            all_radiant_events = all_radiant_events[-max_events:]
                            logging.info(f"[RADIANT EVENTS] START - Limited {original_event_count} events to {len(all_radiant_events)} most recent (max_count_events={max_events})")
                        
                        event_only_message.add_event(all_radiant_events)
                        self.__messages.add_message(event_only_message)
                        logging.info(f'[RADIANT EVENTS] STARTUP - Radiant conversation started with {len(all_radiant_events)} events')

                        # Clear the radiant event log after events are successfully used at conversation start
                        logging.info(f"[RADIANT EVENTS] CLEANUP - Clearing {len(all_radiant_events)} used events after conversation start")
                        self.__context.clear_radiant_event_log()
                        # Reset processed count since log is now empty
                        self.__processed_radiant_events_count = 0

                        # Also clear regular events since they've been processed into radiant events
                        self.__context.clear_context_ingame_events()
                    else:
                        logging.debug("[RADIANT EVENTS] STARTUP - No events available for radiant conversation start")
            else:
                self.__conversation_type.adjust_existing_message_thread(new_prompt, self.__messages)
                self.__messages.reload_message_thread(new_prompt, self.__llm_client.is_too_long, self.TOKEN_LIMIT_RELOAD_MESSAGES)

    @utils.time_it
    def update_game_events(self, message: UserMessage) -> UserMessage:
        """Add in-game events to player's response"""

        all_ingame_events = self.__context.get_context_ingame_events()
        if self.__is_player_interrupting:
            all_ingame_events.append('Interrupting...')
            self.__is_player_interrupting = False
        max_events = min(len(all_ingame_events) ,self.__context.config.max_count_events)
        message.add_event(all_ingame_events[-max_events:])
        self.__context.clear_context_ingame_events()        

        if message.count_ingame_events() > 0:            
            logging.log(28, f'In-game events since previous exchange:\n{message.get_ingame_events_text()}')

        return message

    @utils.time_it
    def retrieve_sentence_from_queue(self) -> Sentence | None:
        """Retrieves the next sentence from the queue.
        If there is a sentence, adds the sentence to the last assistant_message of the message_thread.
        If the last message is not an assistant_message, a new one will be added.

        Returns:
            sentence | None: The next sentence from the queue or None if the queue is empty
        """
        next_sentence: Sentence | None = self.__sentences.get_next_sentence() #This is a blocking call. Execution will wait here until queue is filled again
        if not next_sentence:
            return None
        
        if not next_sentence.is_system_generated_sentence and not next_sentence.speaker.is_player_character:
            last_message = self.__messages.get_last_message()
            if not isinstance(last_message, AssistantMessage):
                last_message = AssistantMessage(self.__context.config)
                last_message.is_multi_npc_message = self.__context.npcs_in_conversation.contains_multiple_npcs()
                self.__messages.add_message(last_message)
            last_message.add_sentence(next_sentence)
            # Clear regular context events after NPC speaks (for player conversations)
            if not isinstance(self.__conversation_type, radiant):
                self.__context.clear_context_ingame_events()
            else:
                # For radiant conversations, check for new events that happened during the conversation
                current_time = time.time()
                
                # Check if radiant conversation duration has been exceeded FIRST
                if not self.__radiant_end_requested and self.__context.config.max_radiant_duration > 0:
                    elapsed_time = current_time - self.__radiant_start_time
                    if elapsed_time >= self.__context.config.max_radiant_duration:
                        logging.info(f"[RADIANT] Duration limit reached ({elapsed_time:.1f}s >= {self.__context.config.max_radiant_duration}s) - requesting end prompt")
                        self.__conversation_type.request_end_prompt()
                        self.__radiant_end_requested = True
                        # DON'T clear the radiant event log - events will be saved when conversation ends
                        # Just stop processing new events
                        logging.info(f"[RADIANT] Stopping all event and vision injection due to time limit")
                        logging.info(f"[RADIANT] Preserving {len(self.__context.get_radiant_event_log())} unprocessed events for next conversation")
                
                # Only check for events and vision if conversation hasn't been requested to end
                if not self.__radiant_end_requested:
                    current_radiant_events = self.__context.get_radiant_event_log()
                    new_events_count = len(current_radiant_events) - self.__processed_radiant_events_count

                    # Check if we should trigger periodic vision
                    # FIXED: Only trigger periodic vision if enough time has passed since BOTH:
                    # 1. The last time we triggered a vision capture
                    # 2. The last time we used a vision result in the LLM
                    should_trigger_vision = False
                    if self.__context.config.periodic_vision_interval > 0 and self.__last_vision_trigger_time > 0:
                        time_since_last_trigger = current_time - self.__last_vision_trigger_time
                        time_since_last_used = current_time - self.__last_vision_used_time if self.__last_vision_used_time > 0 else 999999
                        
                        # Only trigger if BOTH intervals have passed (prevents rapid vision calls)
                        if (time_since_last_trigger >= self.__context.config.periodic_vision_interval and
                            time_since_last_used >= self.__context.config.periodic_vision_interval):
                            should_trigger_vision = True
                            logging.debug(f"[RADIANT VISION] PERIODIC - Triggering (last trigger: {time_since_last_trigger:.1f}s ago, last used: {time_since_last_used:.1f}s ago)")

                    # Inject events and vision
                    if new_events_count > 0 or should_trigger_vision:
                        events_to_process = []

                        if new_events_count > 0:
                            # There are new events that happened during this conversation
                            new_events = current_radiant_events[-new_events_count:]  # Get only the new events
                            
                            # Apply max_count_events limit - only send most recent events
                            max_events = self.__context.config.max_count_events
                            
                            if max_events > 0 and len(new_events) > max_events:
                                # Keep only the most recent events
                                events_to_send = new_events[-max_events:]
                                logging.info(f"[RADIANT EVENTS] DURING - Limited {len(new_events)} new events to {len(events_to_send)} most recent (max_count_events={max_events})")
                            else:
                                events_to_send = new_events
                            
                            events_to_process.extend(events_to_send)
                            logging.info(f"[RADIANT EVENTS] DURING - Sending {len(events_to_send)} events to LLM")
                            
                            # Clear the radiant event log after sending to LLM to start fresh accumulation
                            self.__context.clear_radiant_event_log()
                            # Reset processed count since log is now empty
                            self.__processed_radiant_events_count = 0
                            logging.info(f"[RADIANT EVENTS] DURING - Cleared event log after sending to LLM, will accumulate fresh events")

                        if should_trigger_vision:
                            # Ensure vision injection is allowed for periodic updates
                            self.__allow_vision_injection = True
                            # Trigger periodic vision
                            vision_description = self.__trigger_periodic_vision()
                            if vision_description:
                                vision_event = f"Updated visual context as seen by MJR (the player): {vision_description}"
                                events_to_process.append(vision_event)
                                # Mark this vision as used immediately since we're adding it to the current LLM request
                                self.__last_vision_used_time = current_time
                            self.__last_vision_trigger_time = current_time

                        if events_to_process:
                            logging.debug(f"[RADIANT EVENTS] DURING - Queueing {len(events_to_process)} events for batched injection")
                            
                            # Use debounce mechanism to batch with other simultaneous injections (e.g., late vision)
                            self.__queue_injection(events_to_process)
                # Removed redundant event logging for performance during radiant conversations
        return next_sentence
   
    @utils.time_it
    def initiate_end_sequence(self):
        """Replaces all remaining sentences with a "goodbye" sentence that also prompts the game to request a stop to the conversation using an action

        Also cleans up any pending vision requests for this conversation."""
        # Clear vision requested flag when conversation ends
        self.__vision_requested_for_current_conversation = False

        # Clear the late vision callback if it exists
        if hasattr(self.__llm_client, '_image_client') and self.__llm_client._image_client:
            self.__llm_client._image_client.set_late_vision_callback(None)

        if not self.__has_already_ended:
            config = self.__context.config
            self.__stop_generation()
            self.__sentences.clear()
            if self.__stt:
                self.__stt.stop_listening()
                self.__allow_mic_input = False
            # say goodbyes
            npc = self.__context.npcs_in_conversation.last_added_character
            if npc:
                goodbye_sentence = self.__output_manager.generate_sentence(SentenceContent(npc, config.goodbye_npc_response, SentenceTypeEnum.SPEECH, True))
                if goodbye_sentence:
                    goodbye_sentence.actions.append(comm_consts.ACTION_ENDCONVERSATION)
                    self.__sentences.put(goodbye_sentence)
                    
    @utils.time_it
    def contains_character(self, ref_id: str) -> bool:
        for actor in self.__context.npcs_in_conversation.get_all_characters():
            if actor.ref_id == ref_id:
                return True
        return False
    
    @utils.time_it
    def get_character(self, ref_id: str) -> Character | None:
        for actor in self.__context.npcs_in_conversation.get_all_characters():
            if actor.ref_id == ref_id:
                return actor
        return None

    @utils.time_it
    def end(self):
        """Ends a conversation
        """
        self.__has_already_ended = True
        self.__stop_generation()
        self.__sentences.clear()
        # Reset vision timers when conversation ends
        self.__last_vision_trigger_time = 0
        self.__last_vision_used_time = 0
        # Clear vision callback when conversation ends
        if hasattr(self.__llm_client, '_image_client') and self.__llm_client._image_client:
            self.__llm_client._image_client.set_late_vision_callback(None)
        
        # Save conversation in background thread to avoid blocking (especially for radiant conversations)
        if isinstance(self.__conversation_type, radiant):
            save_thread = Thread(target=self.__save_conversation, args=(False,), name=f"RadiantSave-{id(self)}")
            save_thread.daemon = False  # FIXED: Non-daemon ensures save completes even on exit
            save_thread.start()
        else:
            # For player conversations, keep synchronous to ensure data is saved before player moves on
            self.__save_conversation(is_reload=False)
    
    @utils.time_it
    def __queue_injection(self, events: list[str]):
        """Queue events/vision for debounced injection to prevent duplicate LLM calls
        
        Args:
            events: List of event strings to inject
        """
        with self.__pending_injection_lock:
            # Add events to pending queue
            self.__pending_injection_events.extend(events)
            
            # Cancel existing timer if one is running
            if self.__pending_injection_timer and self.__pending_injection_timer.is_alive():
                # Timer is already running, events will be included in that batch
                logging.debug(f"[RADIANT INJECTION] Added {len(events)} events to existing batch (total: {len(self.__pending_injection_events)})")
                return
            
            # Start new debounce timer
            self.__pending_injection_timer = Thread(target=self.__process_pending_injections_after_delay, daemon=True)
            self.__pending_injection_timer.start()
            logging.debug(f"[RADIANT INJECTION] Started debounce timer for {len(self.__pending_injection_events)} events")
    
    def __process_pending_injections_after_delay(self):
        """Wait for debounce period, then inject all accumulated events in a single LLM call"""
        time.sleep(self.__injection_debounce_seconds)
        
        with self.__pending_injection_lock:
            if not self.__pending_injection_events:
                return  # All events were already processed
            
            # Grab all pending events
            events_to_inject = self.__pending_injection_events.copy()
            self.__pending_injection_events.clear()
            
            logging.info(f"[RADIANT INJECTION] Injecting batched {len(events_to_inject)} events after {self.__injection_debounce_seconds}s debounce")
        
        # IMPORTANT: Check if generation is currently running
        # If it is, wait for it to complete before injecting
        if self.__generation_thread and self.__generation_thread.is_alive():
            logging.debug("[RADIANT INJECTION] Waiting for current generation to complete before injecting events")
            # Wait for current generation to finish (with timeout to avoid infinite wait)
            wait_start = time.time()
            while self.__generation_thread and self.__generation_thread.is_alive():
                if time.time() - wait_start > 30:  # 30 second timeout
                    logging.warning("[RADIANT INJECTION] Timeout waiting for generation to complete, injecting anyway")
                    break
                time.sleep(0.5)
            logging.debug("[RADIANT INJECTION] Previous generation completed, now injecting events")
        
        # Inject outside the lock to avoid blocking other injections
        if not self.__has_already_ended:
            new_events_message = UserMessage(self.__context.config, "", "", False)
            new_events_message.add_event(events_to_inject)
            self.__messages.add_message(new_events_message)
            
            # Now start generation with all batched events
            self.__start_generating_npc_sentences()
    
    @utils.time_it
    def __start_generating_npc_sentences(self):
        """Starts a background Thread to generate sentences into the SentenceQueue"""    
        with self.__generation_start_lock:
            # Check if generation is already running to prevent duplicate calls
            if self.__generation_thread and self.__generation_thread.is_alive():
                logging.debug("[LLM] Generation already in progress, skipping duplicate start request")
                return
            
            if not self.__generation_thread:
                self.__sentences.is_more_to_come = True
                self.__generation_thread = Thread(None, self.__output_manager.generate_response, None, [self.__messages, self.__context.npcs_in_conversation, self.__sentences, self.context.config.actions]).start()   

    @utils.time_it
    def __stop_generation(self):
        """Stops the current generation of sentences if there is one
        """
        self.__output_manager.stop_generation()
        # Reduced polling frequency to minimize CPU overhead
        while self.__generation_thread and self.__generation_thread.is_alive():
            time.sleep(0.05)
        self.__generation_thread = None

    @utils.time_it
    def __prepare_eject_npc_from_conversation(self, npc: Character):
        if not self.__has_already_ended:            
            self.__stop_generation()
            self.__sentences.clear()            
            # say goodbye
            goodbye_sentence = self.__output_manager.generate_sentence(SentenceContent(npc, self.__context.config.goodbye_npc_response, SentenceTypeEnum.SPEECH, False))
            if goodbye_sentence:
                goodbye_sentence.actions.append(comm_consts.ACTION_REMOVECHARACTER)
                self.__sentences.put(goodbye_sentence)        

    @utils.time_it
    def __save_conversation(self, is_reload: bool):
        """Saves conversation log and state for each NPC in the conversation"""
        self.__save_conversations_for_characters(self.__context.npcs_in_conversation.get_all_characters(), is_reload)

    @utils.time_it
    def __save_conversations_for_characters(self, characters_to_save_for: list[Character], is_reload: bool):
        characters_object = Characters()
        for npc in characters_to_save_for:
            if not npc.is_player_character:
                characters_object.add_or_update_character(npc)
                conversation_log.save_conversation_log(npc, self.__messages.transform_to_openai_messages(self.__messages.get_talk_only()), self.__context.world_id)
        
        # Get event log for radiant conversations
        event_log = None
        if isinstance(self.__conversation_type, radiant):
            event_log = self.__context.get_radiant_event_log()
            if event_log and len(event_log) > 0:
                logging.info(f"[RADIANT SUMMARY] Passing {len(event_log)} events to summary: {event_log}")
        
        self.__rememberer.save_conversation_state(self.__messages, characters_object, self.__context.world_id, is_reload, event_log)
        
        # Clear radiant event log after it's been saved to the summary
        if isinstance(self.__conversation_type, radiant) and event_log and len(event_log) > 0:
            self.__context.clear_radiant_event_log()
            logging.info(f"[RADIANT SUMMARY] Cleared {len(event_log)} events from radiant log after saving to summary")

    @utils.time_it
    def __initiate_reload_conversation(self):
        """Places a "gather thoughts" sentence add the front of the queue that also prompts the game to request a reload of the conversation using an action"""
        latest_npc = self.__context.npcs_in_conversation.last_added_character
        if not latest_npc: 
            self.initiate_end_sequence()
            return
        
        # Play gather thoughts
        collecting_thoughts_text = self.__context.config.collecting_thoughts_npc_response
        collecting_thoughts_sentence = self.__output_manager.generate_sentence(SentenceContent(latest_npc, collecting_thoughts_text, SentenceTypeEnum.SPEECH, True))
        if collecting_thoughts_sentence:
            collecting_thoughts_sentence.actions.append(comm_consts.ACTION_RELOADCONVERSATION)
            self.__sentences.put_at_front(collecting_thoughts_sentence)
    
    @utils.time_it
    def reload_conversation(self):
        """Reloads the conversation
        """
        self.__save_conversation(is_reload=True)
        # Reload
        new_prompt = self.__conversation_type.generate_prompt(self.__context)
        self.__messages.reload_message_thread(new_prompt, self.__llm_client.is_too_long, self.TOKEN_LIMIT_RELOAD_MESSAGES)

    @utils.time_it
    def __has_conversation_ended(self, last_user_text: str) -> bool:
        """Checks if the last player text has ended the conversation

        Args:
            last_user_text (str): the text to check

        Returns:
            bool: true if the conversation has ended, false otherwise
        """
        # transcriber = self.__stt
        config = self.__context.config
        transcript_cleaned = utils.clean_text(last_user_text)

        # check if user is ending conversation
        return Transcriber.activation_name_exists(transcript_cleaned, self.__end_conversation_keywords)

    @utils.time_it
    def __does_dismiss_npc_from_conversation(self, last_user_text: str) -> Character | None:
        """Checks if the last player text dismisses an NPC from the conversation

        Args:
            last_user_text (str): the text to check

        Returns:
            bool: true if the conversation has ended, false otherwise
        """
        transcript_cleaned = utils.clean_text(last_user_text)

        words = transcript_cleaned.split()
        for i, word in enumerate(words):
            if word in self.__end_conversation_keywords:
                if i < (len(words) - 1):
                    next_word = words[i + 1]
                    for npc_name in self.__context.npcs_in_conversation.get_all_names():
                        if next_word in npc_name.lower().split():
                            return self.__context.npcs_in_conversation.get_character_by_name(npc_name)
        return None
    
    @utils.time_it
    def __should_voice_player_input(self, player_character: Character) -> bool:
        game_value: Any = player_character.get_custom_character_value(comm_consts.KEY_ACTOR_PC_VOICEPLAYERINPUT)
        if game_value == None:
            return self.__context.config.voice_player_input
        return game_value