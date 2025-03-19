desired_tokens = 10_000_000_000
sequence_len = 4096
gradient_accumulation_steps = 2
micro_batch_size = 8
num_gpus = 1
sample_packing = True  # Set this to False if not using sample packing

# Define the effective sequence length when using sample packing
# This value should be obtained from preprocessing statistics
if sample_packing:
    effective_sequence_len = 3.227e4  # Example value, update based on actual data
else:
    effective_sequence_len = sequence_len

tokens_per_step = effective_sequence_len * gradient_accumulation_steps * micro_batch_size * num_gpus
max_steps = desired_tokens // tokens_per_step

print(f"Set max_steps in config to: {max_steps}")
print(f"10pct: {max_steps // 10}")
