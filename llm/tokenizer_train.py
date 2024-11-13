# Train a tokenizer
import os

import sentencepiece as spm

vocab_size = 4096
# Purpose: You have a lot of datasets that is text data. But neural network cannot work with text. It needs numbers.
# Soln. Text -> tokens -> numbers
# tokens = set(elements that allow you to encode your data), this depends on vocab size.
# higher the vocab_size -> higher the precision in output

# Process(text -> tokens) -> use alog. (Byte Pair Encoding BPE)
# Use (chars of language) -> Encode (all data) |
# repeat until vocab_size is reach
#   [Lookup (chars, its freq). -> Replace (char(highest freq.), token) -> set<TOKEN>.add(token)]


if __name__ == "__main__":
    spm.SentencePieceTrainer.Train(
        input='wiki.txt',
        model_prefix="test_wiki_tokenizer",
        model_type="bpe",  # name of the encoding algo.
        vocab_size=vocab_size,
        self_test_sample_size=0,
        input_format="text",
        character_coverage=0.995,
        # specifies the proportion of the characters in the training corpus that are considered
        # when you build tokenizer model. It means that the tokenizer is going to include the most frequent 99.5% of
        # characters in the training corpus. The remaining 0.5% of the less freq. characters will be treated unknown,
        # this help in managing vocab size and ensure that tokenizer focuses on  the most common characters which
        # improves efficiency and performance.
        num_threads=os.cpu_count(),
        split_digits=True,
        allow_whitespace_only_pieces=True,
        byte_fallback=True,
        # agar koi particular subword token nahi bhi milta trained vocab. mein, jo text hoga wo phir
        # bhi tokenize hoga byte-level representation mein.
        unk_surface=r" \342\201\207 ",
        normalization_rule_name="identity"
    )
    print("Tokenizer training completed...")
