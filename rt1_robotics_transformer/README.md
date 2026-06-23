# Minimal RT-1 in PyTorch

This is a compact PyTorch reimplementation of the core RT-1 ideas for learning and experimentation.

It intentionally keeps the model small and readable:

- FiLM-conditioned CNN image encoder
- TokenLearner-style visual token compression
- Temporal Transformer over frame tokens
- Discrete action prediction with one vocabulary per action dimension

It does **not** try to faithfully reproduce:

- the original EfficientNet tokenizer
- the exact Universal Sentence Encoder pipeline
- real robot data loading
- distributed training and large-scale infrastructure

## Files

- `rt1_model.py`: model definition and action discretization helpers
- `train_demo.py`: synthetic dataset plus a tiny training loop

## RT-1 idea map

```text
images (6 frames) + text instruction
    -> visual encoder with FiLM conditioning
    -> TokenLearner compresses visual tokens
    -> Transformer integrates time
    -> action tokens (discrete bins for each action dimension)
```

## Quick start

From this folder:

```bash
python train_demo.py
```

You should see the loss print every 10 steps and then one example prediction.

## Tensor shapes

- images: `[batch, time, 3, H, W]`
- text embedding: `[batch, text_dim]`
- compressed tokens per frame: `tokens_per_frame`
- transformer input: `[batch, time * tokens_per_frame, embed_dim]`
- output logits: `[batch, num_action_dims, vocab_size]`

## How this maps to the paper

This version keeps the same high-level structure as RT-1:

1. Turn each image frame into a set of visual tokens.
2. Condition the visual stack on language.
3. Compress visual tokens to keep inference cheap.
4. Model temporal context with a Transformer.
5. Predict discretized robot actions with classification losses.

For actual research reproduction, the next upgrades would be:

- replace the toy CNN with a stronger vision backbone
- replace synthetic text embeddings with a real language encoder
- use a real trajectory dataset
- add causal masking and autoregressive action prediction if desired
