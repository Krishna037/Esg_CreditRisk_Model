"""
models.py
---------
Centralized model loader for:
  - BERT        (bert-base-uncased via HuggingFace Transformers)
  - FinBERT     (ProsusAI/finbert via HuggingFace Transformers)
  - Ollama      (local LLM via the `ollama` Python client)

Usage:
    from models import load_bert, load_finbert, query_ollama

Prerequisites:
    pip install transformers torch ollama
    # Ollama desktop app: https://ollama.com/download
    # Then: ollama pull llama3
"""

import torch
from transformers import (
    BertTokenizer,
    BertModel,
    AutoTokenizer,
    AutoModelForSequenceClassification,
    pipeline,
)

# Ollama is optional — import gracefully so the rest of the module works
# even if the Ollama server is not installed or not running.
try:
    import ollama as _ollama
    _OLLAMA_AVAILABLE = True
except ImportError:
    _ollama = None
    _OLLAMA_AVAILABLE = False
    print(
        "[Ollama] WARNING: 'ollama' Python package not found.\n"
        "         Run: pip install ollama\n"
        "         Then install the desktop app from https://ollama.com/download"
    )


# ──────────────────────────────────────────────
# BERT (bert-base-uncased)
# ──────────────────────────────────────────────

def load_bert(model_name: str = "bert-base-uncased"):
    """
    Load a vanilla BERT model and tokenizer from HuggingFace.

    Returns:
        tokenizer : BertTokenizer
        model     : BertModel  (in eval mode)
    """
    print(f"[BERT] Loading tokenizer and model: {model_name} ...")
    tokenizer = BertTokenizer.from_pretrained(model_name)
    model = BertModel.from_pretrained(model_name)
    model.eval()
    print("[BERT] Loaded successfully.\n")
    return tokenizer, model


def bert_encode(text: str, tokenizer, model):
    """
    Encode a string with BERT and return the [CLS] token embedding.

    Args:
        text      : Input string
        tokenizer : BertTokenizer instance
        model     : BertModel instance

    Returns:
        cls_embedding : torch.Tensor of shape (1, hidden_size)
    """
    inputs = tokenizer(
        text,
        return_tensors="pt",
        truncation=True,
        max_length=512,
        padding=True,
    )
    with torch.no_grad():
        outputs = model(**inputs)
    cls_embedding = outputs.last_hidden_state[:, 0, :]  # [CLS] token
    return cls_embedding


# ──────────────────────────────────────────────
# FinBERT (ProsusAI/finbert)
# ──────────────────────────────────────────────

FINBERT_MODEL_NAME = "ProsusAI/finbert"

def load_finbert(model_name: str = FINBERT_MODEL_NAME):
    """
    Load FinBERT for financial sentiment classification.

    Labels: positive | negative | neutral

    Returns:
        tokenizer : AutoTokenizer
        model     : AutoModelForSequenceClassification (in eval mode)
        pipe      : HuggingFace pipeline for quick inference
    """
    print(f"[FinBERT] Loading tokenizer and model: {model_name} ...")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSequenceClassification.from_pretrained(model_name)
    model.eval()

    pipe = pipeline(
        "text-classification",
        model=model,
        tokenizer=tokenizer,
        device=0 if torch.cuda.is_available() else -1,
    )
    print("[FinBERT] Loaded successfully.\n")
    return tokenizer, model, pipe


def finbert_sentiment(text: str, pipe) -> dict:
    """
    Run financial sentiment inference on a piece of text.

    Args:
        text : Input string (max ~512 tokens)
        pipe : HuggingFace pipeline returned by load_finbert()

    Returns:
        dict with keys 'label' and 'score'
    """
    result = pipe(text, truncation=True, max_length=512)
    return result[0]  # e.g. {'label': 'positive', 'score': 0.97}


# ──────────────────────────────────────────────
# Ollama (local LLM)
# ──────────────────────────────────────────────

OLLAMA_DEFAULT_MODEL = "llama3"   # Change to any model you have pulled locally

def query_ollama(
    prompt: str,
    model: str = OLLAMA_DEFAULT_MODEL,
    system: str = "You are a helpful financial analysis assistant.",
) -> str:
    """
    Send a prompt to a locally-running Ollama model and return the response.

    Prerequisites:
        1. Install the Ollama desktop app: https://ollama.com/download
        2. Start it (it runs as a background service on port 11434)
        3. Pull the model you want:
               ollama pull llama3

    Args:
        prompt : User prompt / question
        model  : Ollama model name (default: llama3)
        system : System message for the model

    Returns:
        str : Model's text response
    """
    if not _OLLAMA_AVAILABLE:
        raise RuntimeError(
            "[Ollama] 'ollama' Python package is not installed.\n"
            "         Fix: /usr/local/bin/python3 -m pip install ollama"
        )

    print(f"[Ollama] Querying model '{model}' ...")
    try:
        response = _ollama.chat(
            model=model,
            messages=[
                {"role": "system", "content": system},
                {"role": "user",   "content": prompt},
            ],
        )
        reply = response["message"]["content"]
        print("[Ollama] Response received.\n")
        return reply
    except Exception as e:
        raise RuntimeError(
            f"[Ollama] Could not connect to the Ollama server: {e}\n"
            "         Make sure the Ollama desktop app is running,\n"
            "         then run:  ollama pull llama3"
        ) from e


# ──────────────────────────────────────────────
# Quick sanity-check (run this file directly)
# ──────────────────────────────────────────────

if __name__ == "__main__":
    sample_text = (
        "The company reported record revenues this quarter, "
        "with strong growth across all business segments."
    )

    # --- BERT ---
    bert_tokenizer, bert_model = load_bert()
    embedding = bert_encode(sample_text, bert_tokenizer, bert_model)
    print(f"[BERT]    CLS embedding shape : {embedding.shape}\n")

    # --- FinBERT ---
    _, _, finbert_pipe = load_finbert()
    sentiment = finbert_sentiment(sample_text, finbert_pipe)
    print(f"[FinBERT] Sentiment result    : {sentiment}\n")

    # --- Ollama ---
    # Uncomment after confirming Ollama is running locally:
    # reply = query_ollama("Summarise the ESG risks for a steel manufacturer.")
    # print(f"[Ollama]  Reply:\n{reply}")
