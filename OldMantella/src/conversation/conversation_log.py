import json
import logging
import os
from pathlib import Path
import sys
import tempfile
import time
from threading import Lock
import src.utils as utils
from src.character_manager import Character
from openai.types.chat import ChatCompletionMessageParam

class conversation_log:
    game_path: str = "" # <- This gets set in the __init__ of gameable. Not clean but cleaner than other options
    _file_locks: dict[str, Lock] = {}  # Per-file locks to prevent race conditions
    _locks_lock: Lock = Lock()  # Lock for the locks dictionary

    @staticmethod
    @utils.time_it
    def save_conversation_log(character: Character, messages: list[ChatCompletionMessageParam], world_id: str):
        # save conversation history
        if len(messages) == 0:
            return
        
        conversation_history_file = conversation_log.__get_path_to_conversation_history_file(character, world_id)
        
        # Get or create file-specific lock to prevent race conditions
        with conversation_log._locks_lock:
            if conversation_history_file not in conversation_log._file_locks:
                conversation_log._file_locks[conversation_history_file] = Lock()
            file_lock = conversation_log._file_locks[conversation_history_file]
        
        # Acquire file-specific lock - prevents simultaneous writes
        with file_lock:
            # Load existing conversation history
            if os.path.exists(conversation_history_file):
                try:
                    with open(conversation_history_file, 'r', encoding='utf-8') as f:
                        conversation_history = json.load(f)
                except json.JSONDecodeError as e:
                    # FIXED: This block is now INSIDE the except clause
                    logging.warning(f"Corrupted conversation history file detected for {character.name}: {conversation_history_file}")
                    logging.warning(f"JSON decode error: {e}")
                    # Backup the corrupted file with timestamp
                    backup_file = conversation_history_file + f'.corrupted.{int(time.time())}'
                    try:
                        os.rename(conversation_history_file, backup_file)
                        logging.info(f"Backed up corrupted file to: {backup_file}")
                    except (PermissionError, OSError) as rename_error:
                        logging.error(f"Failed to backup corrupted file: {rename_error}")
                    # Start fresh with current conversation
                    conversation_history = []
            else:
                # First conversation for this character
                directory = os.path.dirname(conversation_history_file)
                os.makedirs(directory, exist_ok=True)
                conversation_history = []
            
            # Append new conversation
            conversation_history.append(messages)
            
            # FIXED: Atomic write using temp file + rename
            # This prevents corruption if process crashes during write
            temp_fd, temp_path = tempfile.mkstemp(
                dir=os.path.dirname(conversation_history_file),
                suffix='.tmp',
                text=False
            )
            try:
                # Write to temp file first
                with os.fdopen(temp_fd, 'w', encoding='utf-8') as f:
                    json.dump(conversation_history, f, indent=4, ensure_ascii=False)
                
                # Atomic rename - only succeeds if write completed
                os.replace(temp_path, conversation_history_file)
                logging.debug(f"Saved conversation log for {character.name} ({len(conversation_history)} conversations)")
            except Exception as e:
                logging.error(f"Failed to save conversation log for {character.name}: {e}")
                # Clean up temp file on error
                try:
                    if os.path.exists(temp_path):
                        os.unlink(temp_path)
                except:
                    pass
                raise

    @staticmethod   
    @utils.time_it 
    def load_conversation_log(character: Character, world_id: str) -> list[str]:
        conversation_history_file = conversation_log.__get_path_to_conversation_history_file(character, world_id)
        if os.path.exists(conversation_history_file):
            try:
                with open(conversation_history_file, 'r', encoding='utf-8') as f:
                    conversation_history = json.load(f)
                previous_conversations = []
                for conversation in conversation_history:
                    previous_conversations.extend(conversation)
                return previous_conversations
            except json.JSONDecodeError as e:
                logging.warning(f"Corrupted conversation history file detected for {character.name}: {conversation_history_file}")
                logging.warning(f"JSON decode error: {e}")
                # Backup the corrupted file (file handle is already closed by with block)
                backup_file = conversation_history_file + '.corrupted'
                try:
                    # If backup already exists, remove it first
                    if os.path.exists(backup_file):
                        try:
                            os.remove(backup_file)
                        except (PermissionError, OSError):
                            # Can't remove old backup, use timestamped backup instead
                            import time
                            backup_file = conversation_history_file + f'.corrupted.{int(time.time())}'
                    os.rename(conversation_history_file, backup_file)
                    logging.info(f"Backed up corrupted file to: {backup_file}")
                except (PermissionError, OSError) as rename_error:
                    logging.error(f"Failed to backup corrupted file: {rename_error}")
                    logging.info(f"Continuing without backup - will return empty list - corrupted file remains in place")
                return []
        else:
            return []

    @staticmethod
    @utils.time_it
    def get_conversation_log_length(character: Character, world_id: str) -> int:
        conversation_history_file = conversation_log.__get_path_to_conversation_history_file(character, world_id)
        if os.path.exists(conversation_history_file):
            try:
                with open(conversation_history_file, 'r', encoding='utf-8') as f:
                    conversation_history = json.load(f)
                return sum(len(conversation) for conversation in conversation_history)
            except json.JSONDecodeError as e:
                logging.warning(f"Corrupted conversation history file detected for {character.name}: {conversation_history_file}")
                logging.warning(f"JSON decode error: {e}")
                # Backup the corrupted file (file handle is already closed by with block)
                backup_file = conversation_history_file + '.corrupted'
                try:
                    # If backup already exists, remove it first
                    if os.path.exists(backup_file):
                        try:
                            os.remove(backup_file)
                        except (PermissionError, OSError):
                            # Can't remove old backup, use timestamped backup instead
                            import time
                            backup_file = conversation_history_file + f'.corrupted.{int(time.time())}'
                    os.rename(conversation_history_file, backup_file)
                    logging.info(f"Backed up corrupted file to: {backup_file}")
                except (PermissionError, OSError) as rename_error:
                    logging.error(f"Failed to backup corrupted file: {rename_error}")
                    logging.info(f"Continuing without backup - will return 0 for conversation length - corrupted file remains in place")
                return 0
        else:
            return 0

    @staticmethod    
    @utils.time_it
    def __get_path_to_conversation_history_file(character: Character, world_id: str) -> str:
        # if multiple NPCs in a conversation have the same name (eg Whiterun Guard) their names are appended with number IDs
        # these IDs need to be removed when saving the conversation
        name: str = utils.remove_trailing_number(character.name)
        non_ref_path = f"{conversation_log.game_path}/{world_id}/{name}/{name}.json"
        ref_path = f"{conversation_log.game_path}/{world_id}/{name} - {character.ref_id}/{name}.json"

        if os.path.exists(non_ref_path): # if a conversation folder already exists for this NPC, use it
            return non_ref_path
        else: # else include the NPC's reference ID in the folder name to differentiate generic NPCs
            return ref_path
