from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
from langchain_huggingface import HuggingFacePipeline
import torch


def load_llm(
    model_id: str = "TinyLlama/TinyLlama-1.1B-Chat-v1.0",
    max_new_tokens: int = 256,
    temperature: float = 0.3,
    use_gpu: bool = torch.cuda.is_available()
) -> HuggingFacePipeline:
    """
    Loads a HuggingFace causal LLM and wraps it in a LangChain-compatible interface.

    Args:
        model_id: HuggingFace model ID.
        max_new_tokens: Max tokens to generate.
        temperature: Sampling temperature.
        use_gpu: If True, uses CUDA if available.

    Returns:
        HuggingFacePipeline compatible with LangChain.
    """
    device = "cuda" if use_gpu and torch.cuda.is_available() else "cpu"
    print(f"🚀 Loading model: {model_id}")
    print(f"🖥️  Device: {device.upper()}")

    try:
        tokenizer = AutoTokenizer.from_pretrained(model_id)

        if device == "cuda":
            model = AutoModelForCausalLM.from_pretrained(
                model_id,
                device_map="auto",  # Let accelerate handle GPU placement
                torch_dtype=torch.float16
            )

            # Confirm GPU details
            current_device = torch.cuda.current_device()
            gpu_name = torch.cuda.get_device_name(current_device)
            print(f"✅ Model is using GPU: {gpu_name}")

        else:
            model = AutoModelForCausalLM.from_pretrained(
                model_id,
                device_map={"": "cpu"},
                torch_dtype=torch.float32
            )

        # Print first parameter's device
        print(f"📦 Model weights are loaded on: {next(model.parameters()).device}")

        # Note: Don't pass 'device' to pipeline if using device_map=auto
        pipe = pipeline(
            "text-generation",
            model=model,
            tokenizer=tokenizer,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            do_sample=False,
            top_p=0.95,
            repetition_penalty=1.1
        )

        return HuggingFacePipeline(pipeline=pipe)

    except Exception as e:
        print(f"❌ Failed to load model: {e}")
        raise RuntimeError("Model loading failed. Check model ID, GPU drivers, or internet connection.")
