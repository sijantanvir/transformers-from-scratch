# transformers-from-scratch

from-scratch implementation of Transformer from the paper, "Attention Is All You Need".

## Structure

```
src/
├── components/          # Standalone building blocks
│   ├── tokenization.py  # word level tokenizer
│   ├── embedding.py     # Token + positional embedding
│   ├── positional_encoding.py
│   ├── attention.py     # Scaled dot-product attention
│   ├── multi_head_attention.py
│   ├── feed_forward.py
│   ├── layer_norm.py
│   └── encoder_block.py
├── model/               # Assembled Transformer
│   ├── decoder_block.py
│   └── transformer.py
├── data/
│   └── dataloader.py     # Custom DataLoader
└── training/
    ├── loss.py
    ├── validate.py        # 3 tasks (forward/backward/optimizer)
    └── trainer.py
```

## Components

- [x] **Custom Data Loader**
- [x] **01 — Tokenization**
- [x] **02 — Embedding Layer**
- [ ] **03 — Positional Encoding**
- [ ] **04 — Scaled Dot-Product Attention**
- [ ] **05 — Multi-Head Attention**
- [ ] **06 — Feed-Forward Network**
- [ ] **07 — Layer Normalization**
- [ ] **08 — Encoder Block**