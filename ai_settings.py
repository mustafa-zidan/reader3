"""
AI settings management for Reader3.
Handles AI provider configuration (LM Studio, Ollama) and chat functionality.
"""

import httpx
import json
import os
from dataclasses import dataclass, field, asdict
from datetime import datetime
from enum import Enum
from typing import List, Dict, Optional, Any


class AIProvider(str, Enum):
    """Supported AI providers."""
    LM_STUDIO = "lm_studio"
    OLLAMA = "ollama"


@dataclass
class AISettings:
    """AI provider configuration."""
    provider: str = AIProvider.LM_STUDIO.value
    server_url: str = "http://localhost:1234/v1"
    model: str = ""
    enabled: bool = False
    temperature: float = 0.7
    max_tokens: int = 2048
    system_prompt: str = "You are a helpful reading assistant. Help the user understand and discuss the text they are reading."
    updated_at: str = field(default_factory=lambda: datetime.now().isoformat())


@dataclass
class ChatMessage:
    """A chat message."""
    id: str
    role: str  # "user" or "assistant"
    content: str
    timestamp: str = field(default_factory=lambda: datetime.now().isoformat())


@dataclass
class ChatSession:
    """A chat session with message history."""
    id: str
    book_id: str
    messages: List[Dict[str, Any]] = field(default_factory=list)
    created_at: str = field(default_factory=lambda: datetime.now().isoformat())
    updated_at: str = field(default_factory=lambda: datetime.now().isoformat())


class AISettingsManager:
    """Manages AI settings and chat sessions."""

    def __init__(self, data_dir: str):
        self.data_dir = data_dir
        self.settings_file = os.path.join(data_dir, "ai_settings.json")
        self.settings: AISettings = AISettings()
        self.chat_sessions: Dict[str, ChatSession] = {}
        self._ensure_dir()
        self.load()

    def _ensure_dir(self):
        """Ensure data directory exists."""
        if not os.path.exists(self.data_dir):
            os.makedirs(self.data_dir)

    def load(self):
        """Load AI settings from file."""
        if os.path.exists(self.settings_file):
            try:
                with open(self.settings_file, 'r', encoding='utf-8') as f:
                    data = json.load(f)

                # Load settings
                if 'settings' in data:
                    self.settings = AISettings(**data['settings'])

                # Load chat sessions
                if 'chat_sessions' in data:
                    for session_id, session_data in data['chat_sessions'].items():
                        self.chat_sessions[session_id] = ChatSession(**session_data)
            except Exception as e:
                print(f"Error loading AI settings: {e}")
                self.settings = AISettings()
                self.chat_sessions = {}

    def save(self):
        """Save AI settings to file."""
        try:
            data = {
                'settings': asdict(self.settings),
                'chat_sessions': {
                    sid: asdict(session) for sid, session in self.chat_sessions.items()
                }
            }
            with open(self.settings_file, 'w', encoding='utf-8') as f:
                json.dump(data, f, indent=2, ensure_ascii=False)
        except Exception as e:
            print(f"Error saving AI settings: {e}")

    def get_settings(self) -> AISettings:
        """Get current AI settings."""
        return self.settings

    def update_settings(self, **kwargs) -> AISettings:
        """Update AI settings."""
        for key, value in kwargs.items():
            if hasattr(self.settings, key):
                setattr(self.settings, key, value)
        self.settings.updated_at = datetime.now().isoformat()
        self.save()
        return self.settings

    def get_default_url(self, provider: str) -> str:
        """Get default server URL for a provider."""
        if provider == AIProvider.LM_STUDIO.value:
            return "http://localhost:1234/v1"
        elif provider == AIProvider.OLLAMA.value:
            return "http://localhost:11434"
        return "http://localhost:1234/v1"

    async def test_connection(self) -> Dict[str, Any]:
        """Test connection to the AI provider."""
        try:
            async with httpx.AsyncClient(timeout=10.0) as client:
                if self.settings.provider == AIProvider.LM_STUDIO.value:
                    # LM Studio uses OpenAI-compatible API
                    response = await client.get(f"{self.settings.server_url}/models")
                    if response.status_code == 200:
                        return {"success": True, "message": "Connected to LM Studio"}
                    else:
                        return {"success": False, "message": f"LM Studio returned status {response.status_code}"}

                elif self.settings.provider == AIProvider.OLLAMA.value:
                    # Ollama API
                    response = await client.get(f"{self.settings.server_url}/api/tags")
                    if response.status_code == 200:
                        return {"success": True, "message": "Connected to Ollama"}
                    else:
                        return {"success": False, "message": f"Ollama returned status {response.status_code}"}

                return {"success": False, "message": "Unknown provider"}
        except httpx.ConnectError:
            return {"success": False,
                    "message": f"Cannot connect to {self.settings.server_url}. Is the server running?"}
        except Exception as e:
            return {"success": False, "message": str(e)}

    async def list_models(self) -> List[Dict[str, str]]:
        """List available models from the provider."""
        try:
            async with httpx.AsyncClient(timeout=10.0) as client:
                if self.settings.provider == AIProvider.LM_STUDIO.value:
                    response = await client.get(f"{self.settings.server_url}/models")
                    if response.status_code == 200:
                        data = response.json()
                        models = []
                        for model in data.get('data', []):
                            models.append({
                                "id": model.get('id', ''),
                                "name": model.get('id', '').split('/')[-1]
                            })
                        return models

                elif self.settings.provider == AIProvider.OLLAMA.value:
                    response = await client.get(f"{self.settings.server_url}/api/tags")
                    if response.status_code == 200:
                        data = response.json()
                        models = []
                        for model in data.get('models', []):
                            models.append({
                                "id": model.get('name', ''),
                                "name": model.get('name', '')
                            })
                        return models

                return []
        except Exception as e:
            print(f"Error listing models: {e}")
            return []

    async def chat(self, messages: List[Dict[str, str]], book_id: str = None) -> Dict[str, Any]:
        """Send a chat message and get a response."""
        if not self.settings.model:
            return {"success": False, "error": "No model selected"}

        try:
            # Prepare messages with system prompt
            full_messages = [
                                {"role": "system", "content": self.settings.system_prompt}
                            ] + messages

            async with httpx.AsyncClient(timeout=60.0) as client:
                if self.settings.provider == AIProvider.LM_STUDIO.value:
                    # LM Studio uses OpenAI-compatible API
                    response = await client.post(
                        f"{self.settings.server_url}/chat/completions",
                        json={
                            "model": self.settings.model,
                            "messages": full_messages,
                            "temperature": self.settings.temperature,
                            "max_tokens": self.settings.max_tokens,
                            "stream": False
                        }
                    )
                    if response.status_code == 200:
                        data = response.json()
                        content = data['choices'][0]['message']['content']
                        return {"success": True, "content": content}
                    else:
                        return {"success": False, "error": f"API error: {response.status_code}"}

                elif self.settings.provider == AIProvider.OLLAMA.value:
                    # Ollama API
                    response = await client.post(
                        f"{self.settings.server_url}/api/chat",
                        json={
                            "model": self.settings.model,
                            "messages": full_messages,
                            "stream": False,
                            "options": {
                                "temperature": self.settings.temperature,
                                "num_predict": self.settings.max_tokens
                            }
                        }
                    )
                    if response.status_code == 200:
                        data = response.json()
                        content = data.get('message', {}).get('content', '')
                        return {"success": True, "content": content}
                    else:
                        return {"success": False, "error": f"API error: {response.status_code}"}

                return {"success": False, "error": "Unknown provider"}
        except httpx.ConnectError:
            return {"success": False, "error": f"Cannot connect to {self.settings.server_url}"}
        except Exception as e:
            return {"success": False, "error": str(e)}

    def get_chat_session(self, book_id: str) -> ChatSession:
        """Get or create a chat session for a book."""
        if book_id not in self.chat_sessions:
            import hashlib
            session_id = hashlib.md5(f"{book_id}-{datetime.now().isoformat()}".encode()).hexdigest()[:12]
            self.chat_sessions[book_id] = ChatSession(
                id=session_id,
                book_id=book_id
            )
            self.save()
        return self.chat_sessions[book_id]

    def add_message(self, book_id: str, role: str, content: str) -> ChatMessage:
        """Add a message to a chat session."""
        import hashlib
        session = self.get_chat_session(book_id)
        msg_id = hashlib.md5(f"{role}-{content}-{datetime.now().isoformat()}".encode()).hexdigest()[:12]
        message = {
            "id": msg_id,
            "role": role,
            "content": content,
            "timestamp": datetime.now().isoformat()
        }
        session.messages.append(message)
        session.updated_at = datetime.now().isoformat()
        self.save()
        return ChatMessage(**message)

    def get_messages(self, book_id: str) -> List[Dict[str, Any]]:
        """Get all messages for a book's chat session."""
        session = self.get_chat_session(book_id)
        return session.messages

    def clear_chat(self, book_id: str):
        """Clear chat history for a book."""
        if book_id in self.chat_sessions:
            self.chat_sessions[book_id].messages = []
            self.chat_sessions[book_id].updated_at = datetime.now().isoformat()
            self.save()
