desired_tokens = 10_000_000_000  # 10 billion tokens
sequence_len = 32768  # 8 * 4096
gradient_accumulation_steps = 16
num_gpus = 1
micro_batch_size = 1

tokens_per_step = sequence_len * gradient_accumulation_steps * micro_batch_size * num_gpus
max_steps = desired_tokens // tokens_per_step

print(f"Set max_steps in config to: {max_steps}")
print(f"10pct: {max_steps // 10}")
