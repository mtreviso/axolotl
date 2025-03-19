from datasets import load_dataset
from transformers import AutoTokenizer

DATASET = "HuggingFaceFW/fineweb-edu"
MODEL = "Qwen/Qwen2.5-7B"
DESIRED_TOKENS = 20_000_000_000  # 20B tokens
SAMPLE_SIZE = 1_000_000  # Adjust as needed for accurate estimation (inf for full dataset)
DATASET_SAMPLE_SIZE = 1_430_000_000  # Total samples available on dataset card


def estimate_tokens_per_sample(dataset, tokenizer):
    total_tokens = 0
    total_samples = 0
    for i, sample in enumerate(dataset):
        tokens = tokenizer(sample["text"])["input_ids"]
        total_tokens += len(tokens)
        total_samples += 1
        if total_samples >= SAMPLE_SIZE:
            break
        if total_tokens >= DESIRED_TOKENS:
            break
        print('Total tokens so far: {} / {} (sample {} / {})'.format(
            total_tokens, DESIRED_TOKENS, total_samples, SAMPLE_SIZE), end='\r')
    return total_tokens, total_samples


if __name__ == "__main__":
    tokenizer = AutoTokenizer.from_pretrained(MODEL, trust_remote_code=True)
    dataset = load_dataset(DATASET, split="train", streaming=True)

    if SAMPLE_SIZE == float('inf'):
        print("Estimating tokens per sample on full dataset...")
        total_tokens, total_samples = estimate_tokens_per_sample(dataset, tokenizer)
        print('Total tokens: {}'.format(total_tokens))

        # compute estimated number of shards
        avg_tokens = total_tokens / total_samples
        print(f"Average tokens per sample: {avg_tokens}")

        shard_num = int(total_tokens / DESIRED_TOKENS)
        print(f"Approximate number of shards: {shard_num}")

    else:
        print("Estimating tokens per sample on full dataset...")
        total_tokens, total_samples = estimate_tokens_per_sample(dataset, tokenizer)
        print('Total tokens: {}'.format(total_tokens))

        avg_tokens = total_tokens / total_samples
        print(f"Average tokens per sample: {avg_tokens}")

        # full_total_samples available on dataset card
        total_estimated_tokens = avg_tokens * DATASET_SAMPLE_SIZE
        shard_num = int(total_estimated_tokens / DESIRED_TOKENS)
        print(f"Approximate number of shards: {shard_num}")
