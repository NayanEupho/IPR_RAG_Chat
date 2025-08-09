# rag_chatbot/LLM_interface.py
"""
LLM interface supporting two backends:
  - Hugging Face local (default)
  - Ollama (local Ollama daemon at http://localhost:11434)

Provides a simple .invoke(prompt) API for both backends so rest of the app
can stay backend-agnostic.
"""

import os
import json
import requests
from typing import Optional
import torch

from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline

# --- HF adapter -------------------------------------------------------------
class HFLocalLLM:
    def __init__(self, model_id: str, max_new_tokens: int = 256, temperature: float = 0.3, use_gpu: bool = False):
        self.model_id = model_id
        self.max_new_tokens = max_new_tokens
        self.temperature = temperature
        self.use_gpu = use_gpu

        print(f"🚀 Loading HF model: {model_id} (use_gpu={use_gpu})")
        tokenizer = AutoTokenizer.from_pretrained(model_id)

        # Load model with device_map to allow auto placement if GPU is requested
        if use_gpu and torch.cuda.is_available():
            model = AutoModelForCausalLM.from_pretrained(
                model_id,
                device_map="auto",
                torch_dtype=torch.float16
            )
            print("✅ HF model loaded with device_map=auto (GPU)")
        else:
            model = AutoModelForCausalLM.from_pretrained(
                model_id,
                device_map={"": "cpu"},
                torch_dtype=torch.float32
            )
            print("✅ HF model loaded on CPU")

        self.pipe = pipeline(
            "text-generation",
            model=model,
            tokenizer=tokenizer,
            max_new_tokens=self.max_new_tokens,
            temperature=self.temperature,
            do_sample=False,
            top_p=0.95,
            repetition_penalty=1.1
        )

    def invoke(self, prompt: str) -> str:
        # use pipeline to generate text; returns a list of dicts
        output = self.pipe(prompt)
        # Attempt to extract text robustly
        if isinstance(output, list) and len(output) > 0:
            candidate = output[0]
            # typical key for transformers text-generation pipeline
            text = candidate.get("generated_text") or candidate.get("text") or ""
            return text.strip()
        return str(output).strip()


# --- Ollama adapter ---------------------------------------------------------
class OllamaLLM:
    """
    Adapter for Ollama local daemon (http://localhost:11434).
    Uses the /api/chat endpoint with streaming disabled to get a JSON response.
    """
    def __init__(self, model: str = "phi3:mini", base_url: str = "http://localhost:11434", timeout: int = 60):
        self.model = model
        self.base_url = base_url.rstrip("/")
        self.timeout = timeout
        print(f"🚀 Ollama adapter configured for model={self.model} at {self.base_url}")

    def invoke(self, prompt: str) -> str:
        """
        Send a chat request to Ollama and return the assistant's text.
        We request stream: false to get a single JSON response.
        """
        url = f"{self.base_url}/api/chat"
        payload = {
            "model": self.model,
            "messages": [{"role": "user", "content": prompt}],
            "stream": False
        }
        try:
            resp = requests.post(url, json=payload, timeout=self.timeout)
            resp.raise_for_status()
            data = resp.json()
            # Ollama responses vary; attempt common fields:
            #  - data.get("choices")[0]["message"]["content"]
            #  - data.get("text")
            if isinstance(data, dict):
                if "choices" in data and isinstance(data["choices"], list) and len(data["choices"]) > 0:
                    choice = data["choices"][0]
                    if isinstance(choice, dict):
                        msg = choice.get("message", {}) or {}
                        content = msg.get("content") or msg.get("text") or None
                        if content:
                            return content.strip()
                # fallback to 'text' or 'result' keys
                if "text" in data and isinstance(data["text"], str):
                    return data["text"].strip()
                # some Ollama responses embed output in 'content' key
                if "content" in data and isinstance(data["content"], str):
                    return data["content"].strip()
            # fallback: return raw body
            return resp.text.strip()
        except Exception as e:
            raise RuntimeError(f"Ollama invocation failed: {e}")


# --- Loader ---------------------------------------------------------------
def load_llm(
    model_id: str = "TinyLlama/TinyLlama-1.1B-Chat-v1.0",
    backend: str = "hf",
    max_new_tokens: int = 256,
    temperature: float = 0.3,
    use_gpu: bool = torch.cuda.is_available()
):
    """
    Unified loader returning an object with .invoke(prompt) that returns a string.
    - backend: 'hf' or 'ollama'
    - model_id: for 'hf' this is HF model id; for 'ollama' this is Ollama model string (e.g. 'phi3:mini')
    """
    backend = (backend or "hf").lower()
    if backend in ("hf", "huggingface"):
        return HFLocalLLM(model_id=model_id, max_new_tokens=max_new_tokens, temperature=temperature, use_gpu=use_gpu)
    elif backend in ("ollama", "ollama_local"):
        # For Ollama, use the model_id directly (e.g. 'phi3:mini')
        base_url = os.environ.get("OLLAMA_BASE_URL", "http://localhost:11434")
        return OllamaLLM(model=model_id, base_url=base_url)
    else:
        raise ValueError(f"Unknown backend: {backend}. Supported: 'hf', 'ollama'")
