"""
Orpheus TTS module for Reader3.
Uses LM Studio with the Orpheus TTS model for text-to-speech.

Based on: https://github.com/isaiahbjork/orpheus-tts-local
Original Orpheus TTS: https://github.com/canopyai/Orpheus-TTS

Requirements:
- LM Studio running locally with Orpheus TTS model loaded
- Install TTS dependencies: pip install reader3[tts]
"""

import io
import json
import os
import wave
from typing import Optional, List

import httpx

# TTS configuration
SAMPLE_RATE = 24000  # SNAC model uses 24kHz
AVAILABLE_VOICES = ["tara", "leah", "jess", "leo", "dan", "mia", "zac", "zoe"]
DEFAULT_VOICE = "tara"

# Default generation parameters
DEFAULT_TEMPERATURE = 0.6
DEFAULT_TOP_P = 0.9
DEFAULT_REPETITION_PENALTY = 1.1
DEFAULT_MAX_TOKENS = 2048

# Token processing constants
CUSTOM_TOKEN_PREFIX = "<custom_token_"

# Try to import SNAC decoder (optional dependency)
_snac_available = False
_snac_model = None
_snac_device = "cpu"

try:
    import torch
    import numpy as np
    from snac import SNAC

    _snac_available = True


    def _init_snac_model():
        """Initialize the SNAC model for audio decoding."""
        global _snac_model, _snac_device

        if _snac_model is not None:
            return _snac_model

        # Determine device
        if torch.cuda.is_available():
            _snac_device = "cuda"
        elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
            _snac_device = "mps"
        else:
            _snac_device = "cpu"

        print(f"Orpheus TTS: Loading SNAC model on {_snac_device}")
        _snac_model = SNAC.from_pretrained("hubertsiuzdak/snac_24khz").eval()
        _snac_model = _snac_model.to(_snac_device)
        return _snac_model


    def _convert_tokens_to_audio(multiframe: List[int], count: int) -> Optional[bytes]:
        """Convert token frames to audio bytes using SNAC decoder."""
        if len(multiframe) < 7:
            return None

        model = _init_snac_model()

        codes_0 = torch.tensor([], device=_snac_device, dtype=torch.int32)
        codes_1 = torch.tensor([], device=_snac_device, dtype=torch.int32)
        codes_2 = torch.tensor([], device=_snac_device, dtype=torch.int32)

        num_frames = len(multiframe) // 7
        frame = multiframe[:num_frames * 7]

        for j in range(num_frames):
            i = 7 * j
            if codes_0.shape[0] == 0:
                codes_0 = torch.tensor([frame[i]], device=_snac_device, dtype=torch.int32)
            else:
                codes_0 = torch.cat([codes_0, torch.tensor([frame[i]], device=_snac_device, dtype=torch.int32)])

            if codes_1.shape[0] == 0:
                codes_1 = torch.tensor([frame[i + 1]], device=_snac_device, dtype=torch.int32)
                codes_1 = torch.cat([codes_1, torch.tensor([frame[i + 4]], device=_snac_device, dtype=torch.int32)])
            else:
                codes_1 = torch.cat([codes_1, torch.tensor([frame[i + 1]], device=_snac_device, dtype=torch.int32)])
                codes_1 = torch.cat([codes_1, torch.tensor([frame[i + 4]], device=_snac_device, dtype=torch.int32)])

            if codes_2.shape[0] == 0:
                codes_2 = torch.tensor([frame[i + 2]], device=_snac_device, dtype=torch.int32)
                codes_2 = torch.cat([codes_2, torch.tensor([frame[i + 3]], device=_snac_device, dtype=torch.int32)])
                codes_2 = torch.cat([codes_2, torch.tensor([frame[i + 5]], device=_snac_device, dtype=torch.int32)])
                codes_2 = torch.cat([codes_2, torch.tensor([frame[i + 6]], device=_snac_device, dtype=torch.int32)])
            else:
                codes_2 = torch.cat([codes_2, torch.tensor([frame[i + 2]], device=_snac_device, dtype=torch.int32)])
                codes_2 = torch.cat([codes_2, torch.tensor([frame[i + 3]], device=_snac_device, dtype=torch.int32)])
                codes_2 = torch.cat([codes_2, torch.tensor([frame[i + 5]], device=_snac_device, dtype=torch.int32)])
                codes_2 = torch.cat([codes_2, torch.tensor([frame[i + 6]], device=_snac_device, dtype=torch.int32)])

        codes = [codes_0.unsqueeze(0), codes_1.unsqueeze(0), codes_2.unsqueeze(0)]

        # Validate token ranges
        if (torch.any(codes[0] < 0) or torch.any(codes[0] > 4096) or
                torch.any(codes[1] < 0) or torch.any(codes[1] > 4096) or
                torch.any(codes[2] < 0) or torch.any(codes[2] > 4096)):
            return None

        with torch.inference_mode():
            audio_hat = model.decode(codes)

        audio_slice = audio_hat[:, :, 2048:4096]
        detached_audio = audio_slice.detach().cpu()
        audio_np = detached_audio.numpy()
        audio_int16 = (audio_np * 32767).astype(np.int16)
        audio_bytes = audio_int16.tobytes()

        return audio_bytes

except ImportError:
    _snac_available = False


    def _init_snac_model():
        raise ImportError("TTS dependencies not installed. Run: pip install reader3[tts]")


    def _convert_tokens_to_audio(multiframe: List[int], count: int) -> Optional[bytes]:
        raise ImportError("TTS dependencies not installed. Run: pip install reader3[tts]")


def is_tts_available() -> bool:
    """Check if TTS dependencies are available."""
    return _snac_available


def get_available_voices() -> List[str]:
    """Get list of available TTS voices."""
    return AVAILABLE_VOICES.copy()


def _format_prompt(text: str, voice: str = DEFAULT_VOICE) -> str:
    """Format prompt for Orpheus model with voice prefix and special tokens."""
    if voice not in AVAILABLE_VOICES:
        print(f"Warning: Voice '{voice}' not recognized. Using '{DEFAULT_VOICE}' instead.")
        voice = DEFAULT_VOICE

    formatted_prompt = f"{voice}: {text}"
    special_start = "<|audio|>"
    special_end = "<|eot_id|>"

    return f"{special_start}{formatted_prompt}{special_end}"


def _turn_token_into_id(token_string: str, index: int) -> Optional[int]:
    """Convert token string to numeric ID for audio processing."""
    token_string = token_string.strip()

    last_token_start = token_string.rfind(CUSTOM_TOKEN_PREFIX)
    if last_token_start == -1:
        return None

    last_token = token_string[last_token_start:]

    if last_token.startswith(CUSTOM_TOKEN_PREFIX) and last_token.endswith(">"):
        try:
            number_str = last_token[14:-1]
            token_id = int(number_str) - 10 - ((index % 7) * 4096)
            return token_id
        except ValueError:
            return None

    return None


async def generate_speech(
        text: str,
        voice: str = DEFAULT_VOICE,
        server_url: str = "http://localhost:1234/v1",
        temperature: float = DEFAULT_TEMPERATURE,
        top_p: float = DEFAULT_TOP_P,
        repetition_penalty: float = DEFAULT_REPETITION_PENALTY,
        max_tokens: int = DEFAULT_MAX_TOKENS,
) -> dict:
    """
    Generate speech from text using Orpheus TTS via LM Studio.
    
    Args:
        text: The text to convert to speech
        voice: Voice to use (tara, leah, jess, leo, dan, mia, zac, zoe)
        server_url: LM Studio server URL
        temperature: Generation temperature
        top_p: Top-p sampling parameter
        repetition_penalty: Repetition penalty
        max_tokens: Maximum tokens to generate
    
    Returns:
        dict with 'success', 'audio_data' (base64), 'duration', 'error'
    """
    if not _snac_available:
        return {
            "success": False,
            "error": "TTS dependencies not installed. Run: pip install reader3[tts]"
        }

    formatted_prompt = _format_prompt(text, voice)

    payload = {
        "model": "orpheus-3b-0.1-ft",
        "prompt": formatted_prompt,
        "max_tokens": max_tokens,
        "temperature": temperature,
        "top_p": top_p,
        "repeat_penalty": repetition_penalty,
        "stream": True
    }

    try:
        # Initialize SNAC model
        _init_snac_model()

        buffer = []
        count = 0
        audio_segments = []

        async with httpx.AsyncClient(timeout=120.0) as client:
            async with client.stream(
                    "POST",
                    f"{server_url}/completions",
                    json=payload,
                    headers={"Content-Type": "application/json"}
            ) as response:
                if response.status_code != 200:
                    return {
                        "success": False,
                        "error": f"LM Studio API error: {response.status_code}"
                    }

                async for line in response.aiter_lines():
                    if not line:
                        continue

                    if line.startswith("data: "):
                        data_str = line[6:]
                        if data_str.strip() == "[DONE]":
                            break

                        try:
                            data = json.loads(data_str)
                            if "choices" in data and len(data["choices"]) > 0:
                                token_text = data["choices"][0].get("text", "")
                                if token_text:
                                    token = _turn_token_into_id(token_text, count)
                                    if token is not None and token > 0:
                                        buffer.append(token)
                                        count += 1

                                        # Convert to audio when we have enough tokens
                                        if count % 7 == 0 and count > 27:
                                            buffer_to_proc = buffer[-28:]
                                            audio_bytes = _convert_tokens_to_audio(buffer_to_proc, count)
                                            if audio_bytes:
                                                audio_segments.append(audio_bytes)
                        except json.JSONDecodeError:
                            continue

        if not audio_segments:
            return {
                "success": False,
                "error": "No audio generated. Check if Orpheus model is loaded in LM Studio."
            }

        # Combine audio segments into WAV
        combined_audio = b"".join(audio_segments)

        # Create WAV file in memory
        wav_buffer = io.BytesIO()
        with wave.open(wav_buffer, "wb") as wav_file:
            wav_file.setnchannels(1)
            wav_file.setsampwidth(2)
            wav_file.setframerate(SAMPLE_RATE)
            wav_file.writeframes(combined_audio)

        wav_data = wav_buffer.getvalue()

        # Calculate duration
        duration = len(combined_audio) / (2 * SAMPLE_RATE)

        # Encode as base64 for transport
        import base64
        audio_base64 = base64.b64encode(wav_data).decode("utf-8")

        return {
            "success": True,
            "audio_data": audio_base64,
            "duration": duration,
            "format": "wav",
            "sample_rate": SAMPLE_RATE
        }

    except httpx.ConnectError:
        return {
            "success": False,
            "error": f"Cannot connect to LM Studio at {server_url}. Is it running?"
        }
    except Exception as e:
        return {
            "success": False,
            "error": str(e)
        }


async def generate_speech_to_file(
        text: str,
        output_path: str,
        voice: str = DEFAULT_VOICE,
        server_url: str = "http://localhost:1234/v1",
        **kwargs
) -> dict:
    """
    Generate speech and save to a WAV file.
    
    Args:
        text: The text to convert to speech
        output_path: Path to save the WAV file
        voice: Voice to use
        server_url: LM Studio server URL
        **kwargs: Additional parameters for generate_speech
    
    Returns:
        dict with 'success', 'path', 'duration', 'error'
    """
    result = await generate_speech(text, voice, server_url, **kwargs)

    if not result.get("success"):
        return result

    try:
        import base64
        audio_data = base64.b64decode(result["audio_data"])

        os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)

        with open(output_path, "wb") as f:
            f.write(audio_data)

        return {
            "success": True,
            "path": output_path,
            "duration": result["duration"]
        }
    except Exception as e:
        return {
            "success": False,
            "error": f"Failed to save audio file: {e}"
        }
