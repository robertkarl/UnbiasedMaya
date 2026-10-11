#pragma once
#include "gguf.h"
#include <stdexcept>
#include <string>
// Metadata-only tokenizer artifact: preserves original model identity, tokens
// and template. Split metadata is removed because this artifact has no weight
// shards.
inline void write_vocab(const std::string &source, const std::string &output) {
  gguf_init_params p{true, nullptr};
  auto *input = gguf_init_from_file(source.c_str(), p);
  if (!input)
    throw std::runtime_error("cannot read GGUF vocabulary metadata");
  auto *target = gguf_init_empty();
  gguf_set_kv(target, input);
  gguf_remove_key(target, "split.no");
  gguf_remove_key(target, "split.count");
  gguf_remove_key(target, "split.tensors.count");
  bool ok = gguf_write_to_file(target, output.c_str(), true);
  gguf_free(target);
  gguf_free(input);
  if (!ok)
    throw std::runtime_error("cannot write tokenizer GGUF");
}
