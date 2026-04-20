import src.utils as utils
import logging
from openai.types.chat import ChatCompletionMessageParam
import unicodedata
import threading
import time
from src.config.config_loader import ConfigLoader
from src.image.image_manager import ImageManager
from src.llm.client_base import ClientBase
from src.llm.messages import ImageMessage

class ImageClient(ClientBase):
    '''Image class to handle LLM vision
    '''
    @utils.time_it
    def __init__(self, config: ConfigLoader, secret_key_file: str, image_secret_key_file: str) -> None:
        self.__config = config    
        self.__custom_vision_model: bool = config.custom_vision_model

        if self.__custom_vision_model: # if using a custom model for vision, load these custom config values
            setup_values = {'api_url': config.vision_llm_api, 'llm': config.vision_llm, 'llm_params': config.vision_llm_params, 'custom_token_count': config.vision_custom_token_count}
        else: # default to base LLM config values
            setup_values = {'api_url': config.llm_api, 'llm': config.llm, 'llm_params': config.llm_params, 'custom_token_count': config.custom_token_count}
        
        super().__init__(**setup_values, secret_key_files=[image_secret_key_file, secret_key_file])

        if self.__custom_vision_model:
            if self._is_local:
                logging.info(f"Running local vision model")
            else:
                logging.log(23, f"Running Mantella with custom vision model '{config.vision_llm}'")

        self.__end_of_sentence_chars = ['.', '?', '!', ';', '。', '？', '！', '；', '：']
        self.__end_of_sentence_chars = [unicodedata.normalize('NFKC', char) for char in self.__end_of_sentence_chars]

        self.__vision_prompt: str = config.vision_prompt.format(game=config.game.display_name)
        self.__detail: str = "low" if config.low_resolution_mode else "high"
        self.__image_transcription_prefix: str = config.image_transcription_prefix
        self.__image_manager: ImageManager | None = ImageManager(config.game, 
                                                config.save_folder, 
                                                config.save_screenshot, 
                                                config.image_quality, 
                                                config.low_resolution_mode, 
                                                config.resize_method, 
                                                config.capture_offset,
                                                config.use_game_screenshots,
                                                config.game_path)
    
    @utils.time_it
    def add_image_to_messages(self, openai_messages: list[ChatCompletionMessageParam], vision_hints: str) -> list[ChatCompletionMessageParam]:
        '''Adds a captured image to the latest user message

        Args:
            openai_messages (list[ChatCompletionMessageParam]): The existing list of messages in the OpenAI format

        Returns:
            list[ChatCompletionMessageParam]: The updated list of messages with the image added
        '''
        image = self.__image_manager.get_image()
        if image is None:
            return openai_messages
        
        if not self.__custom_vision_model:
            # Add the image to the last user message or create a new message if needed
            if openai_messages and openai_messages[-1]['role'] == 'user':
                openai_messages[-1]['content'] = [
                    {"type": "text", "text": openai_messages[-1]['content']},
                    {"type": "image_url", "image_url": {"url":  f"data:image/jpeg;base64,{image}", "detail": self.__detail}}
                ]
            else:
                openai_messages.append({
                    "role": "user",
                    "content": [
                        {"type": "text", "text": vision_hints},
                        {"type": "image_url", "image_url": {"url":  f"data:image/jpeg;base64,{image}", "detail": self.__detail}}
                    ]
                })
        else:
            if len(vision_hints) > 0:
                vision_prompt = f"{self.__vision_prompt}\n{vision_hints}"
            else:
                vision_prompt = self.__vision_prompt
            image_msg_instance = ImageMessage(self.__config, image, vision_prompt, self.__detail, True)
            image_transcription = self.request_call(image_msg_instance)
            if image_transcription:
                last_punctuation = max(image_transcription.rfind(p) for p in self.__end_of_sentence_chars)
                # filter transcription to full sentences
                image_transcription = image_transcription if last_punctuation == -1 else image_transcription[:last_punctuation + 1]

                

                # Prepend the custom prefix to the image transcription
                full_transcription = f"{self.__image_transcription_prefix} {image_transcription}" if self.__image_transcription_prefix else image_transcription
                
                logging.log(23, f"Image transcription: {full_transcription}")

                # Add the image to the last user message or create a new message if needed
                if openai_messages and openai_messages[-1]['role'] == 'user':
                    openai_messages[-1]['content'] = [
                        {"type": "text", "text": f"*{full_transcription}*\n{openai_messages[-1]['content']}"}
                    ]
                else:
                    openai_messages.append({
                        "role": "user",
                        "content": [
                            {"type": "text", "text": f"*{full_transcription}*"}
                        ]
                    })

        return openai_messages
    
    @utils.time_it
    def get_vision_description_with_timeout(self, vision_hints: str = "", timeout_seconds: float | None = None) -> str | None:
        '''Gets a vision description with a timeout for radiant conversations

        Args:
            vision_hints (str): Additional context hints for the vision model
            timeout_seconds (float | None): Maximum time to wait for vision response (uses config.vision_timeout if None)

        Returns:
            str | None: Vision description text or None if unavailable/timeout
        '''
        if timeout_seconds is None:
            timeout_seconds = float(self.__config.vision_timeout)

        if not self.__image_manager:
            logging.log(23, "Image manager not available for vision description")
            return None

        image = self.__image_manager.get_image()
        if image is None:
            logging.log(23, "No image available for vision description")
            return None

        # Use a thread to handle the vision request with timeout
        result_container = {'description': None, 'completed': False}

        def get_vision_description():
            try:
                if len(vision_hints) > 0:
                    vision_prompt = f"{self.__vision_prompt}\n{vision_hints}"
                else:
                    vision_prompt = self.__vision_prompt

                # Sending prompt to vision model
                image_msg_instance = ImageMessage(self.__config, image, vision_prompt, self.__detail, True)

                start_time = time.time()
                image_transcription = self.request_call(image_msg_instance)
                end_time = time.time()

                if image_transcription:
                    # Filter transcription to full sentences
                    last_punctuation = max(image_transcription.rfind(p) for p in self.__end_of_sentence_chars)
                    filtered_transcription = image_transcription if last_punctuation == -1 else image_transcription[:last_punctuation + 1]
                    result_container['description'] = filtered_transcription

                    # Vision response received and will be processed by callback

                    # Check if we have a callback for late vision injection
                    if hasattr(self, '_late_vision_callback') and self._late_vision_callback:
                        try:
                            success = self._late_vision_callback(filtered_transcription)
                            if success:
                                logging.info("[RADIANT VISION] LATE - Vision description successfully injected into ongoing conversation")
                            else:
                                logging.warning("[RADIANT VISION] LATE - Failed to inject vision description into ongoing conversation")
                        except Exception as e:
                            logging.error(f"[RADIANT VISION] LATE - Error calling late vision callback: {e}")
                else:
                    logging.warning("[RADIANT VISION] Vision model returned empty response")
            except Exception as e:
                logging.error(f"[RADIANT VISION] Vision description failed with error: {e}")
                import traceback
                logging.error(f"[RADIANT VISION] Full traceback: {traceback.format_exc()}")
            finally:
                result_container['completed'] = True

        # Start the vision request in a separate thread
        # Starting vision request with timeout
        vision_thread = threading.Thread(target=get_vision_description, daemon=True)
        vision_thread.start()

        # Wait for completion or timeout
        start_time = time.time()
        while not result_container['completed'] and (time.time() - start_time) < timeout_seconds:
            time.sleep(0.1)

        if not result_container['completed']:
            logging.debug(f"[RADIANT VISION] TIMEOUT - Vision description timed out after {timeout_seconds:.1f}s")
            # Don't return yet - the thread might still complete and call the callback
            return None

        final_description = result_container['description']
        return final_description

    def set_late_vision_callback(self, callback: callable) -> None:
        """Set a callback function to be called when vision completes after timeout

        Args:
            callback: Function that takes a vision description string and returns bool
                     indicating if injection was successful
        """
        self._late_vision_callback = callback
        # Late vision callback registered for potential delayed injection