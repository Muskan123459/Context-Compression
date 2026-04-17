#!/usr/bin/env python3
"""Interactive terminal chat with SmolLM (GPU preferred; optional CPU)."""

from __future__ import annotations

import argparse
import sys

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

DEFAULT_MODEL = "HuggingFaceTB/SmolLM3-3B"
MAX_CONTEXT_TOKENS = 16_384  # hard cap — native is 64k but we limit to 16k

_CUDA_MISMATCH_HINT = """\
PyTorch cannot use the NVIDIA GPU on this machine. Common cause: the default
`pip install torch` wheel targets a newer CUDA than your driver supports.

Fix (pick one):
  • Match PyTorch to your driver — uninstall then install a cu12 wheel, e.g.:
      pip uninstall -y torch
      pip install torch --index-url https://download.pytorch.org/whl/cu124
    (Try cu121 if cu124 still fails; see https://pytorch.org/get-started/locally/)
  • Upgrade the NVIDIA driver from https://www.nvidia.com/Download/index.aspx
  • Run on CPU (slow; use a small model):
      python scripts/chat_smollm.py --cpu --model HuggingFaceTB/SmolLM2-360M-Instruct
"""


def _model_input_device(model: torch.nn.Module) -> torch.device:
    return next(model.parameters()).device


def main() -> None:
    parser = argparse.ArgumentParser(description="REPL chat with SmolLM (CUDA or --cpu).")
    parser.add_argument("--model", default=DEFAULT_MODEL, help="HF model id or local path.")
    parser.add_argument(
        "--max-new-tokens",
        type=int,
        default=512,
        help="Maximum tokens to generate per reply.",
    )
    parser.add_argument("--temperature", type=float, default=0.7)
    parser.add_argument("--top-p", type=float, default=0.9)
    parser.add_argument(
        "--cpu",
        action="store_true",
        help="Force CPU (float32). Use a small instruct model unless you like waiting.",
    )
    args = parser.parse_args()

    if args.cpu:
        dev = torch.device("cpu")
        dtype = torch.float32
        print(f"Loading {args.model} on CPU …")
        tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
        model = AutoModelForCausalLM.from_pretrained(
            args.model,
            torch_dtype=dtype,
            device_map=None,
            trust_remote_code=True,
            max_position_embeddings=MAX_CONTEXT_TOKENS,
        ).to(dev)
    elif not torch.cuda.is_available():
        print(_CUDA_MISMATCH_HINT, file=sys.stderr)
        sys.exit(1)
    else:
        dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
        print(f"Loading {args.model} …")
        tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
        model = AutoModelForCausalLM.from_pretrained(
            args.model,
            torch_dtype=dtype,
            device_map="auto",
            trust_remote_code=True,
            max_position_embeddings=MAX_CONTEXT_TOKENS,
        )

    model.eval()

    if tokenizer.pad_token_id is None and tokenizer.eos_token_id is not None:
        tokenizer.pad_token = tokenizer.eos_token

    dev = _model_input_device(model)
    print(f"Ready on {dev}. Empty line or Ctrl+D to exit.\n")

    messages: list[dict[str, str]] = []

    while True:
        try:
            user_text = input("You: ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            break
        if not user_text:
            break

        messages.append({"role": "user", "content": user_text})

        if getattr(tokenizer, "chat_template", None):
            prompt = tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True,
            )
        else:
            prompt = "\n".join(f"{m['role']}: {m['content']}" for m in messages) + "\nassistant:"

        inputs = tokenizer(prompt, return_tensors="pt")
        inputs = {k: v.to(dev) for k, v in inputs.items()}

        prompt_len = inputs["input_ids"].shape[1]
        remaining = MAX_CONTEXT_TOKENS - prompt_len
        if remaining <= 0:
            print("Assistant: [context window full — history too long]\n")
            continue
        max_new = min(args.max_new_tokens, remaining)

        with torch.inference_mode():
            out = model.generate(
                **inputs,
                max_new_tokens=max_new,
                do_sample=True,
                temperature=args.temperature,
                top_p=args.top_p,
                pad_token_id=tokenizer.pad_token_id,
            )

        new_tokens = out[0, inputs["input_ids"].shape[1] :]
        reply = tokenizer.decode(new_tokens, skip_special_tokens=True).strip()
        total_tokens = out.shape[1]
        print(f"Assistant: {reply}\n")
        print(f"[tokens: {total_tokens} / {MAX_CONTEXT_TOKENS}]\n")
        messages.append({"role": "assistant", "content": reply})


if __name__ == "__main__":
    main()
