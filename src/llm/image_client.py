import src.utils as utils
from openai.types.chat import ChatCompletionMessageParam
import unicodedata
import threading
import time
import traceback
from typing import Callable, Optional
from src.config.config_loader import ConfigLoader
from src.image.image_manager import ImageManager
from src.llm.client_base import ClientBase
from src.llm.messages import ImageMessage
from src.model_profile_manager import get_profile_manager

logger = utils.get_logger()


class ImageClient(ClientBase):
    '''Image class to handle LLM vision
    '''
    @utils.time_it
    def __init__(self, config: ConfigLoader) -> None:
        self.__config = config    
        self.__custom_vision_model: bool = config.custom_vision_model

        profile_manager = get_profile_manager()
        if self.__custom_vision_model: # if using a custom model for vision, load these custom config values
            resolved_params = profile_manager.resolve_params(
                service=config.vision_llm_api,
                model=config.vision_llm,
                fallback_params=config.vision_llm_params,
                apply_profile=config.apply_model_profiles,
                log_context="ImageClient(custom)",
            )
            setup_values = {'api_url': config.vision_llm_api, 'llm': config.vision_llm, 'llm_params': resolved_params, 'custom_token_count': config.vision_custom_token_count}
        else: # default to base LLM config values
            resolved_params = profile_manager.resolve_params(
                service=config.llm_api,
                model=config.llm,
                fallback_params=config.llm_params,
                apply_profile=config.apply_model_profiles,
                log_context="ImageClient(base)",
            )
            setup_values = {'api_url': config.llm_api, 'llm': config.llm, 'llm_params': resolved_params, 'custom_token_count': config.custom_token_count}
        
        super().__init__(**setup_values)

        if self.__custom_vision_model:
            if self._is_local:
                logger.info(f"Running local vision model")
            else:
                logger.log(23, f"Running Mantella with custom vision model '{config.vision_llm}'")

        self.__end_of_sentence_chars = ['.', '?', '!', ';', '。', '？', '！', '；', '：']
        self.__end_of_sentence_chars = [unicodedata.normalize('NFKC', char) for char in self.__end_of_sentence_chars]

        self.__vision_prompt: str = config.vision_prompt.format(game=config.game.display_name)
        self.__detail: str = "low" if config.low_resolution_mode else "high"
        self.__image_manager: ImageManager | None = ImageManager(config.game, 
                                                config.save_folder, 
                                                config.save_screenshot, 
                                                config.image_quality, 
                                                config.low_resolution_mode, 
                                                config.resize_method, 
                                                config.capture_offset,
                                                config.use_game_screenshots,
                                                config.game_path)

        # Callback invoked when an asynchronous vision description finishes after the
        # requesting conversation has already moved on (used for radiant NPC-to-NPC
        # conversations where vision should never block the LLM response).
        self._late_vision_callback: Optional[Callable[[str], bool]] = None
    
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
        
        # Find the last user message by walking backwards, skipping assistant/tool messages
        last_user_message_idx = None
        for i in range(len(openai_messages) - 1, -1, -1):
            if openai_messages[i]['role'] == 'user':
                last_user_message_idx = i
                break
        
        if not self.__custom_vision_model:
            # Add the image to the last user message or create a new message if needed
            if last_user_message_idx is not None:
                openai_messages[last_user_message_idx]['content'] = [
                    {"type": "text", "text": openai_messages[last_user_message_idx]['content']},
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

                logger.log(23, f"Image transcription: {image_transcription}")

                # Add the image to the last user message or create a new message if needed
                if last_user_message_idx is not None:
                    openai_messages[last_user_message_idx]['content'] = [
                        {"type": "text", "text": f"*{image_transcription}*\n{openai_messages[last_user_message_idx]['content']}"}
                    ]
                else:
                    openai_messages.append({
                        "role": "user",
                        "content": [
                            {"type": "text", "text": f"*{image_transcription}*"}
                        ]
                    })

        return openai_messages

    @utils.time_it
    def get_vision_description_with_timeout(self, vision_hints: str = "", timeout_seconds: Optional[float] = None) -> Optional[str]:
        """Capture a screenshot and request a text vision description from the vision LLM.

        Unlike ``add_image_to_messages``, this method returns a plain text description
        that can be injected as an in-game event. It is intended for radiant (NPC-to-NPC)
        conversations where vision should never block the normal NPC response.

        Args:
            vision_hints: Additional context hints for the vision model.
            timeout_seconds: Maximum time to wait for the vision response.
                - ``None``: use ``config.vision_timeout``
                - ``0``: fire and forget (returns ``None`` immediately; the callback registered
                  via :meth:`set_late_vision_callback` is invoked when the description is ready)

        Returns:
            The vision description text, or ``None`` if the request timed out or failed.
        """
        if timeout_seconds is None:
            try:
                timeout_seconds = float(self.__config.vision_timeout)
            except Exception:
                timeout_seconds = 0.0

        if not self.__image_manager:
            logger.log(23, "Image manager not available for vision description")
            return None

        image = self.__image_manager.get_image()
        if image is None:
            logger.warning("[RADIANT VISION] No image captured from game window for vision description (window may not be focused/found)")
            return None

        logger.log(23, f"[RADIANT VISION] Image captured ({self.__detail} detail), sending to vision LLM...")

        result_container = {'description': None, 'completed': False}

        def _run_vision_request():
            start_time = time.time()
            try:
                if len(vision_hints) > 0:
                    vision_prompt = f"{self.__vision_prompt}\n{vision_hints}"
                else:
                    vision_prompt = self.__vision_prompt

                image_msg_instance = ImageMessage(self.__config, image, vision_prompt, self.__detail, True)
                image_transcription = self.request_call(image_msg_instance)

                elapsed = time.time() - start_time
                if image_transcription:
                    last_punctuation = max(image_transcription.rfind(p) for p in self.__end_of_sentence_chars)
                    filtered_transcription = image_transcription if last_punctuation == -1 else image_transcription[:last_punctuation + 1]
                    result_container['description'] = filtered_transcription

                    logger.log(24, f"[RADIANT VISION] Vision LLM responded in {elapsed:.1f}s ({len(filtered_transcription)} chars):\n    {filtered_transcription}")

                    callback = self._late_vision_callback
                    if callback:
                        try:
                            callback(filtered_transcription)
                        except Exception as cb_err:
                            logger.error(f"[RADIANT VISION] Late vision callback failed: {cb_err}")
                    else:
                        logger.log(23, "[RADIANT VISION] No late-vision callback registered; description discarded")
                else:
                    logger.warning(f"[RADIANT VISION] Vision model returned empty response after {elapsed:.1f}s")
            except Exception as e:
                logger.error(f"[RADIANT VISION] Vision description request failed: {e}")
                logger.debug(traceback.format_exc())
            finally:
                result_container['completed'] = True

        vision_thread = threading.Thread(target=_run_vision_request, daemon=True, name="RadiantVisionRequest")
        vision_thread.start()

        if timeout_seconds is None or timeout_seconds <= 0:
            # Fire-and-forget: do not block the caller. Description (if any) is delivered
            # through the late vision callback registered via set_late_vision_callback.
            return None

        start_time = time.time()
        while not result_container['completed'] and (time.time() - start_time) < timeout_seconds:
            time.sleep(0.1)

        if not result_container['completed']:
            logger.debug(f"[RADIANT VISION] Vision description timed out after {timeout_seconds:.1f}s (will fall back to late callback)")
            return None

        return result_container['description']

    def set_late_vision_callback(self, callback: Optional[Callable[[str], bool]]) -> None:
        """Register (or clear) a callback invoked when an async vision description arrives.

        Pass ``None`` to clear the callback (e.g. when a radiant conversation ends).
        """
        self._late_vision_callback = callback
