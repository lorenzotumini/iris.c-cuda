#!/bin/bash
set -e

show_usage() {
    echo "Iris Model Downloader"
    echo ""
    echo "Usage: $0 MODEL [--variant bf16] [--token TOKEN] [--output-dir DIR]"
    echo ""
    echo "Available models:"
    echo ""
    echo "  4b            Distilled 4B (4 steps, fast, ~16 GB disk)"
    echo "  4b-base       Base 4B (50 steps, CFG, higher quality, ~16 GB disk)"
    echo "  9b            Distilled 9B (4 steps, higher quality, non-commercial, ~30 GB disk)"
    echo "  9b-base       Base 9B (50 steps, CFG, highest quality, non-commercial, ~30 GB disk)"
    echo "  zimage-turbo  Z-Image-Turbo 6B (8 NFE / 9 scheduler steps, Apache 2.0)"
    echo ""
    echo "Z-Image variants:"
    echo "  fp32           Official main snapshot (~31 GB total, default)"
    echo "  bf16           Native BF16 transformer (~20 GB total, CUDA recommended)"
    echo ""
    echo "By default this implementation uses mmap() so inference is often"
    echo "possible with less RAM than the model size."
    echo ""
    echo "If this is your first time, we suggest downloading the \"4b\" model:"
    echo "  $0 4b"
    exit 1
}

# Need at least one argument
if [ $# -lt 1 ]; then
    show_usage
fi

# First positional argument is the model name
MODEL="$1"
shift

# Map model name to repo and output directory
case "$MODEL" in
    4b)
        REPO="FLUX.2-klein-4B"
        DEFAULT_OUT="./flux-klein-4b"
        ;;
    4b-base)
        REPO="FLUX.2-klein-base-4B"
        DEFAULT_OUT="./flux-klein-4b-base"
        ;;
    9b)
        REPO="FLUX.2-klein-9B"
        DEFAULT_OUT="./flux-klein-9b"
        ;;
    9b-base)
        REPO="FLUX.2-klein-base-9B"
        DEFAULT_OUT="./flux-klein-9b-base"
        ;;
    zimage-turbo)
        ORG="Tongyi-MAI"
        REPO="Z-Image-Turbo"
        DEFAULT_OUT="./zimage-turbo"
        ;;
    *)
        echo "Unknown model: $MODEL"
        echo ""
        show_usage
        ;;
esac

# Parse remaining arguments
TOKEN=""
VARIANT="fp32"
OUT=""
while [ $# -gt 0 ]; do
    case "$1" in
        --token)
            if [ $# -lt 2 ]; then echo "Error: --token requires a value"; exit 1; fi
            TOKEN="$2"
            shift
            ;;
        --variant)
            if [ $# -lt 2 ]; then echo "Error: --variant requires a value"; exit 1; fi
            VARIANT="$2"
            shift
            ;;
        --output-dir|-o)
            if [ $# -lt 2 ]; then echo "Error: --output-dir requires a value"; exit 1; fi
            OUT="$2"
            shift
            ;;
        *)
            echo "Unknown option: $1"
            echo ""
            show_usage
            ;;
    esac
    shift
done

if [ "$VARIANT" != "fp32" ] && [ "$VARIANT" != "bf16" ]; then
    echo "Error: --variant must be fp32 or bf16"
    exit 1
fi
if [ "$MODEL" != "zimage-turbo" ] && [ "$VARIANT" != "fp32" ]; then
    echo "Error: --variant is currently supported only for zimage-turbo"
    exit 1
fi

REVISION="main"
if [ "$MODEL" = "zimage-turbo" ] && [ "$VARIANT" = "bf16" ]; then
    REVISION="refs/pr/102"
    if [ -z "$OUT" ]; then OUT="./zimage-turbo-bf16"; fi
fi
if [ -z "$OUT" ]; then OUT="$DEFAULT_OUT"; fi

# Try to find token from environment
if [ -z "$TOKEN" ] && [ -n "$HF_TOKEN" ]; then
    TOKEN="$HF_TOKEN"
fi

if [ -n "$TOKEN" ]; then
    echo "Using authentication token"
fi

if [ "$MODEL" = "zimage-turbo" ]; then
    echo "Downloading $REPO ($VARIANT)..."
else
    echo "Downloading $REPO..."
fi
echo "Revision: $REVISION"
echo "Output dir: $OUT"

ORG="${ORG:-black-forest-labs}"
REVISION_URL="${REVISION//\//%2F}"
BASE="https://huggingface.co/$ORG/$REPO/resolve/$REVISION_URL"

# Helper function to download with optional auth
curl_file() {
    if [ -n "$TOKEN" ]; then
        curl -fL -H "Authorization: Bearer $TOKEN" -o "$1" "$2"
    else
        curl -fL -o "$1" "$2"
    fi
}

dl() {
    if ! curl_file "$1" "$2"; then
        rm -f "$1"
        echo ""
        echo "Error: failed to download $(basename "$1")"
        echo "URL: $2"
        echo ""
        if [ -z "$TOKEN" ]; then
            echo "This may be a gated model that requires authentication."
            echo "  1. Accept the license at https://huggingface.co/$ORG/$REPO"
            echo "  2. Get your token from https://huggingface.co/settings/tokens"
            echo "  3. Run: $0 $MODEL --token YOUR_TOKEN"
            echo "  Or set the HF_TOKEN env var"
        else
            echo "Authentication failed (HTTP 403). Possible causes:"
            echo "  - Token may be invalid or expired"
            echo "  - You may need to accept the license first:"
            echo "    https://huggingface.co/$ORG/$REPO"
            echo "  - The repository name may not exist (check spelling)"
        fi
        exit 1
    fi
}

# Tokenizer metadata differs between repositories. These files improve
# interoperability when present but Iris does not require all of them.
dl_optional() {
    if ! curl_file "$1" "$2"; then
        rm -f "$1"
        echo "Skipping optional file: $(basename "$1")"
    fi
}

mkdir -p "$OUT"/{text_encoder,tokenizer,transformer,vae}

# model_index.json (needed for autodetection)
dl "$OUT/model_index.json" "$BASE/model_index.json"

# text_encoder (Qwen3 - ~8GB for 4B, ~16GB for 9B)
dl "$OUT/text_encoder/config.json" "$BASE/text_encoder/config.json"
dl "$OUT/text_encoder/generation_config.json" "$BASE/text_encoder/generation_config.json"
dl "$OUT/text_encoder/model.safetensors.index.json" "$BASE/text_encoder/model.safetensors.index.json"

# Discover and download all safetensors shards from the index
SHARDS=$(python3 -c '
import json
import sys
try:
    with open(sys.argv[1]) as f:
        idx = json.load(f)
    shards = sorted(set(idx["weight_map"].values()))
    for s in shards:
        print(s)
except (OSError, KeyError, ValueError):
    # Fallback: assume 2 shards
    print("model-00001-of-00002.safetensors")
    print("model-00002-of-00002.safetensors")
' "$OUT/text_encoder/model.safetensors.index.json" 2>/dev/null)

for shard in $SHARDS; do
    dl "$OUT/text_encoder/$shard" "$BASE/text_encoder/$shard"
done

# tokenizer
dl_optional "$OUT/tokenizer/added_tokens.json" "$BASE/tokenizer/added_tokens.json"
dl_optional "$OUT/tokenizer/chat_template.jinja" "$BASE/tokenizer/chat_template.jinja"
dl "$OUT/tokenizer/merges.txt" "$BASE/tokenizer/merges.txt"
dl_optional "$OUT/tokenizer/special_tokens_map.json" "$BASE/tokenizer/special_tokens_map.json"
dl "$OUT/tokenizer/tokenizer.json" "$BASE/tokenizer/tokenizer.json"
dl "$OUT/tokenizer/tokenizer_config.json" "$BASE/tokenizer/tokenizer_config.json"
dl "$OUT/tokenizer/vocab.json" "$BASE/tokenizer/vocab.json"

# transformer
dl "$OUT/transformer/config.json" "$BASE/transformer/config.json"

# Try the selected transformer's index first, then fall back to its single
# file form. Never fall back from a requested BF16 variant to FP32.
if [ "$MODEL" = "zimage-turbo" ] && [ "$VARIANT" = "bf16" ]; then
    TF_INDEX_NAME="diffusion_pytorch_model.safetensors.index.bf16.json"
    TF_SINGLE_NAME="diffusion_pytorch_model.bf16.safetensors"
else
    TF_INDEX_NAME="diffusion_pytorch_model.safetensors.index.json"
    TF_SINGLE_NAME="diffusion_pytorch_model.safetensors"
fi
TF_INDEX="$OUT/transformer/$TF_INDEX_NAME"
if ! curl_file "$TF_INDEX" "$BASE/transformer/$TF_INDEX_NAME" 2>/dev/null; then
    rm -f "$TF_INDEX"
fi

if [ -f "$TF_INDEX" ]; then
    # Sharded: discover and download all shards
    TF_SHARDS=$(python3 -c '
import json
import sys
with open(sys.argv[1]) as f:
    idx = json.load(f)
shards = sorted(set(idx["weight_map"].values()))
for s in shards:
    print(s)
' "$TF_INDEX" 2>/dev/null)
    for shard in $TF_SHARDS; do
        dl "$OUT/transformer/$shard" "$BASE/transformer/$shard"
    done
else
    # Single file (4B distilled/base, or a future single-file BF16 variant)
    dl "$OUT/transformer/$TF_SINGLE_NAME" "$BASE/transformer/$TF_SINGLE_NAME"
fi

# vae (~168 MB)
dl "$OUT/vae/config.json" "$BASE/vae/config.json"
dl "$OUT/vae/diffusion_pytorch_model.safetensors" "$BASE/vae/diffusion_pytorch_model.safetensors"

echo "Done. -> $OUT"
