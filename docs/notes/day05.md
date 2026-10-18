# Day 5 notes

## Prompting comparison (llama3.1, temperature 0, 10 items)
| strategy | prompt_id | correct | avg completion tokens | avg seconds |
|---|---|---|---|---|
| zero_shot | ... | ./10 | . | . |
| few_shot | ... | ./10 | . | . |
| cot | ... | ./10 | . | . |

Observation: ...
Caveat: n=10 is too small to conclude anything; day 10+ builds a real eval.

## Context window (num_ctx)
- Default behavior on this machine: estimated ___ tokens, Ollama processed ___.
- Answer without num_ctx: ...
- With num_ctx 8192: processed ___, answer ...
- Conclusion: Ollama truncates silently; set num_ctx explicitly and keep RC_LLM_CONTEXT_WINDOW in sync.
- tiktoken (cl100k_base) estimate vs Ollama count: ratio ___.

