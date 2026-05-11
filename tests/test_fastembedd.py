#!/usr/bin/env python
"""
NeuralCore - Standalone FastEmbed download test
Tests the exact default model + init logic used in ContextManager._init_fastembed
Clean, modular, no domain-specific logic.
"""

import asyncio
import time
from fastembed import TextEmbedding
import os


async def test_fastembed_download():
    """Test FastEmbed initialization + model download with default config."""
    print("🚀 Testing FastEmbed download with default model")
    print("=" * 60)

    # Exact defaults from ContextManager._init_fastembed
    model_name = "BAAI/bge-small-en-v1.5"
    start_time = time.time()

    try:
        print(f"📥 Initializing TextEmbedding with model: {model_name}")
        print("   (This will download the model on first run)")

        embedder = TextEmbedding(
            model_name=model_name,
            # No local_path / cache overrides — using pure defaults for test
        )

        # Quick smoke test: embed a tiny batch
        test_texts = ["Hello, this is a test sentence for embedding."]
        embeddings = list(embedder.embed(test_texts))

        elapsed = time.time() - start_time

        print("✅ SUCCESS: FastEmbed model loaded and ready")
        print(f"   Model name : {model_name}")
        print(f"   Embedding dim : {len(embeddings[0])}")
        print(f"   Download + init time : {elapsed:.2f} seconds")
        print(f"   Cache location : {os.getenv('FASTEMBED_CACHE_DIR', '~/.cache/huggingface/hub')}")

        # Optional: show first few values
        print(f"   First 5 values : {embeddings[0][:5].tolist()}")

    except Exception as e:
        print(f"❌ FAILED: {type(e).__name__}: {e}")
        raise

    print("=" * 60)
    print("FastEmbed download test completed successfully ✅")


if __name__ == "__main__":
    asyncio.run(test_fastembed_download())