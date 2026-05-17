"""
Shared model asset download filters.

The patterns approximate the minimal files needed by common from_pretrained
and hub pipeline loaders while avoiding examples, documentation, and media.
"""

MODEL_ALLOW_PATTERNS: tuple[str, ...] = (
    "*.safetensors", "*.bin", "*.pt", "*.pth", "*.ckpt", "*.onnx", "*.tflite", "*.gguf",
    "*.json", "*.yaml", "*.yml", "*.txt", "*.model", "*.spm", "*.bpe", "*.jinja",
    "*.py",
    "config.json", "generation_config.json", "model_index.json",
    "tokenizer.json", "tokenizer_config.json", "special_tokens_map.json",
    "vocab.json", "vocab.txt", "merges.txt",
    "preprocessor_config.json", "processor_config.json", "feature_extractor_config.json",
    "unet/*.json", "vae/*.json", "text_encoder/*.json", "text_encoder_2/*.json",
    "scheduler/*.json",
)

MODEL_IGNORE_PATTERNS: tuple[str, ...] = (
    "*.png", "*.jpg", "*.jpeg", "*.gif", "*.bmp", "*.webp", "*.svg",
    "*.mp4", "*.mov", "*.avi", "*.mkv", "*.webm",
    "*.wav", "*.mp3", "*.flac", "*.ogg",
    "*.md", "*.rst", "*.pdf",
    "assets/**", "figures/**", "images/**", "media/**",
    "docs/**", "doc/**",
    "demo/**", "demos/**", "example/**", "examples/**", "samples/**", "sample/**",
    "tests/**", "test/**", "benchmark/**", "benchmarks/**",
    "training/**", "train/**",
)
