// Native GLM pack writer. Format and precision policy derived from Project Maya
// tools/iq_pack.py (MIT). Experts remain byte-for-byte in their GGUF shards.
#include "ggml-impl.h"
#include "ggml.h"
#include "strata/artifact/gguf_reader.hpp"
#include "vocab.hpp"
#include <cmath>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <limits>
#include <memory>
#include <regex>
#include <sstream>
namespace fs = std::filesystem;
static void require(bool v, const std::string &why) {
  if (!v)
    throw std::runtime_error(why);
}
static bool ends(const std::string &n,
                 std::initializer_list<const char *> names) {
  for (auto s : names)
    if (n.ends_with(s))
      return true;
  return false;
}
static bool bf16(const std::string &n) {
  return n.starts_with("blk.") && ends(n, {"hc_attn_fn.weight",
                                           "hc_attn_base.weight",
                                           "hc_attn_scale.weight",
                                           "hc_ffn_fn.weight",
                                           "hc_ffn_base.weight",
                                           "hc_ffn_scale.weight",
                                           "indexer.attn_q_b.weight",
                                           "indexer.attn_k.weight",
                                           "indexer_compressor_gate.weight",
                                           "indexer_compressor_ape.weight",
                                           "indexer.proj.weight",
                                           "indexer.k_norm.weight",
                                           "indexer.k_norm.bias",
                                           "ffn_gate_inp.weight",
                                           "exp_probs_b.bias",
                                           "ssm_a",
                                           "ssm_dt.bias",
                                           "ssm_norm.weight",
                                           "ssm_beta.weight",
                                           "ssm_f_a.weight",
                                           "ssm_f_b.weight",
                                           "ssm_g_a.weight",
                                           "ssm_g_b.weight",
                                           "ssm_conv1d_q.weight",
                                           "ssm_conv1d_k.weight",
                                           "ssm_conv1d_v.weight",
                                           "attn_q_a_norm.weight",
                                           "attn_kv_a_norm.weight",
                                           "attn_k_b.weight",
                                           "attn_v_b.weight"});
}
static bool native(const std::string &n) {
  return ends(n, {"attn_q_a.weight", "attn_q_b.weight", "attn_kv_a_mqa.weight",
                  "attn_output.weight", "attn_q.weight", "attn_k.weight",
                  "attn_v.weight", "ffn_gate.weight", "ffn_up.weight",
                  "ffn_down.weight", "ffn_gate_shexp.weight",
                  "ffn_up_shexp.weight", "ffn_down_shexp.weight",
                  "token_embd.weight", "output.weight"});
}
struct Item {
  strata::GgufFile *file;
  const strata::TensorInfo *tensor;
  fs::path path;
  uint64_t bytes;
};
static uint64_t size(const strata::TensorInfo &t) {
  require(!t.shape.empty() && t.shape.size() <= 4,
          "invalid dimensions: " + t.name);
  uint64_t n = 1;
  for (auto d : t.shape) {
    require(d && n <= UINT64_MAX / d, "dimension overflow: " + t.name);
    n *= d;
  }
  int block = 0, bytes = 0;
  require(strata::block_geometry(t.type, block, bytes) && n % block == 0 &&
              t.shape[0] % block == 0,
          "unsupported tensor layout: " + t.name);
  require(n / block <= UINT64_MAX / (unsigned)bytes, "tensor size overflow");
  return n / block * bytes;
}
static void run(fs::path first, fs::path out) {
  first = fs::absolute(first);
  out = fs::absolute(out);
  require(!fs::exists(out), "output exists; choose a new pack directory");
  require(out.parent_path() == first.parent_path(),
          "pack must be directly beside the GGUF shards");
  std::vector<fs::path> paths;
  std::smatch m;
  auto name = first.filename().string();
  if (std::regex_search(name, m,
                        std::regex("-([0-9]{5})-of-([0-9]{5})\\.gguf$"))) {
    require(m[1] == "00001", "pass the first shard");
    int count = std::stoi(m[2]);
    require(count > 0 && count < 1000, "invalid shard count");
    for (int i = 1; i <= count; i++) {
      char suffix[40];
      std::snprintf(suffix, sizeof suffix, "-%05d-of-%05d.gguf", i, count);
      paths.push_back(first.parent_path() / (m.prefix().str() + suffix));
    }
  } else
    paths.push_back(first);
  std::vector<std::unique_ptr<strata::GgufFile>> files;
  std::vector<Item> items;
  std::map<std::string, size_t> where;
  for (auto &p : paths) {
    require(p.filename().string().find_first_of(" \t\r\n") == std::string::npos,
            "shard names must contain no whitespace");
    auto f = std::make_unique<strata::GgufFile>(p.string());
    for (auto &t : f->tensors()) {
      auto bytes = size(t);
      auto start = f->data_start();
      require(start <= f->file_size() && t.offset <= f->file_size() - start &&
                  bytes <= f->file_size() - start - t.offset,
              "truncated tensor: " + t.name);
      require(where.emplace(t.name, items.size()).second,
              "duplicate tensor: " + t.name);
      items.push_back({f.get(), &t, p, bytes});
    }
    files.push_back(std::move(f));
  }
  auto *arch = files[0]->get("general.architecture");
  require(arch && (arch->s == "glm5-next" || arch->s == "glm5next"),
          "expected GLM5-Next GGUF");
  auto meta = [&](const char *key, int fallback) {
    auto *v = files[0]->get(arch->s + "." + key);
    return v ? int(v->u) : fallback;
  };
  int trunk = meta("block_count", 0) - meta("nextn_predict_layers", 0),
      lead = meta("leading_dense_block_count", 0);
  require(trunk > lead && lead >= 0, "invalid trunk/leading layer count");
  auto get = [&](int layer, const std::string &role) -> const Item & {
    auto n = "blk." + std::to_string(layer) + "." + role;
    require(where.count(n), "missing tensor: " + n);
    return items[where.at(n)];
  };
  auto &router = *get(lead, "ffn_gate_inp.weight").tensor;
  require(router.shape.size() == 2 && router.shape[1] > 0, "invalid router");
  auto ne = router.shape[1];
  auto tmp = out.string() + ".tmp-" + std::to_string(getpid());
  require(fs::create_directory(tmp), "temporary pack already exists");
  // On failure retain the temporary output for inspection; never replace a
  // valid pack.
  std::ofstream dense(fs::path(tmp) / "dense.bin", std::ios::binary),
      index(fs::path(tmp) / "index.txt"),
      experts(fs::path(tmp) / "native_experts.txt");
  dense.exceptions(std::ios::badbit | std::ios::failbit);
  index.exceptions(std::ios::badbit | std::ios::failbit);
  experts.exceptions(std::ios::badbit | std::ios::failbit);
  std::vector<std::string> rows;
  uint64_t at = 0;
  unsigned converted = 0;
  for (auto &it : items) {
    auto &t = *it.tensor;
    auto &n = t.name;
    if (n.starts_with("blk.")) {
      auto dot = n.find('.', 4);
      require(dot != std::string::npos, "bad layer name");
      auto layer = std::stoi(n.substr(4, dot - 4));
      if (layer >= trunk || n.ends_with("_exps.weight"))
        continue;
    }
    auto dims = t.shape.size();
    if (dims == 4 && t.shape[3] == 1)
      --dims;
    require(dims <= 3, "unsupported dense dimensions: " + n);
    uint64_t n0 = t.shape[0], n1 = dims > 1 ? t.shape[1] : 0;
    if (dims == 3) {
      n0 *= n1;
      n1 = t.shape[2];
    }
    bool convert = bf16(n) && t.type != 30,
         floating = t.type == 0 || t.type == 1 || t.type == 30;
    if (floating || convert) {
      std::vector<uint16_t> rounded;
      const auto *raw = it.file->tensor_data(t);
      auto bytes = it.bytes;
      int kind = t.type == 30 ? 4 : t.type == 1 ? 5 : 2;
      if (convert) {
        const auto *traits = ggml_get_type_traits((ggml_type)t.type);
        uint64_t count = t.elements();
        std::vector<float> values(count);
        if (t.type == 0)
          std::memcpy(values.data(), raw, bytes);
        else {
          require(traits && traits->to_float, "missing dequantizer: " + n);
          traits->to_float(raw, values.data(), count);
        }
        rounded.resize(count);
        for (uint64_t i = 0; i < count; i++) {
          require(std::isfinite(values[i]), "nonfinite BF16 weight: " + n);
          uint32_t u;
          std::memcpy(&u, &values[i], 4);
          rounded[i] = (uint16_t)((u + 0x7fff + ((u >> 16) & 1)) >> 16);
        }
        raw = (const uint8_t *)rounded.data();
        bytes = count * 2;
        kind = 4;
        ++converted;
      }
      rows.push_back(n + " 0 " + std::to_string(kind) + " " +
                     std::to_string(at) + " " + std::to_string(bytes) + " " +
                     std::to_string(at) + " " + std::to_string(bytes) + " " +
                     std::to_string(n0) + " " + std::to_string(n1) +
                     " 0 0 1 0 0 0 0 0 0 0");
      dense.write((const char *)raw, bytes);
      char zero[64] = {};
      auto pad = (64 - bytes % 64) % 64;
      dense.write(zero, pad);
      at += bytes + pad;
    } else {
      require(native(n), "unsupported quantized dense tensor: " + n);
      rows.push_back(n + " 0 0 0 0 " + std::to_string(at) + " 0 " +
                     std::to_string(n0) + " " + std::to_string(n1) +
                     " 8 0 32 0 0 0 0 0 0 0");
    }
  }
  index << "# strata pack index v3 -- generated by maya-pack from " << name
        << "\n# align 64 pool " << at << " tensors " << rows.size() << "\n";
  for (auto &r : rows)
    index << r << '\n';
  std::vector<std::string> erows;
  uint64_t offset = 0;
  for (int l = lead; l < trunk; l++) {
    auto &r = *get(l, "ffn_gate_inp.weight").tensor;
    require(r.shape.size() == 2 && r.shape[1] == ne,
            "router expert counts differ");
    const Item *ts[] = {&get(l, "ffn_gate_exps.weight"),
                        &get(l, "ffn_up_exps.weight"),
                        &get(l, "ffn_down_exps.weight")};
    for (auto t : ts)
      require(t->tensor->shape.size() == 3 && t->tensor->shape[2] == ne &&
                  t->bytes % ne == 0,
              "invalid expert geometry");
    require(ts[0]->tensor->shape == ts[1]->tensor->shape &&
                ts[0]->tensor->type == ts[1]->tensor->type,
            "gate/up mismatch");
    auto blob = (ts[0]->bytes + ts[1]->bytes + ts[2]->bytes) / ne;
    std::ostringstream row;
    row << l << ' ' << ts[0]->tensor->type << ' ' << ts[2]->tensor->type << ' '
        << offset << ' ' << blob;
    for (auto t : ts)
      row << ' ' << t->file->data_start() + t->tensor->offset;
    for (auto t : ts)
      row << ' ' << t->path.filename().string();
    erows.push_back(row.str());
    offset += blob * ne;
  }
  experts
      << "# strata native experts v4: layer gu_type d_type offset blob_bytes "
         "gate_off up_off down_off [gate_shard up_shard down_shard] (n_expert "
      << ne << ", total " << offset << "; absolute offsets in " << name
      << ", or in the named shard(s) beside it)\n";
  for (auto &r : erows)
    experts << r << '\n';
  dense.close();
  index.close();
  experts.close();
  std::ofstream manifest(fs::path(tmp) / "preparation.txt");
  manifest.exceptions(std::ios::badbit | std::ios::failbit);
  manifest << "format=maya-pack-v1\nrounding=BF16-nearest-even\nsource=" << name
           << "\ntrunk=" << trunk << "\nconverted=" << converted << "\n";
  manifest.close();
  write_vocab(first.string(), (fs::path(tmp) / "tokenizer.gguf").string());
  fs::rename(tmp, out);
  std::cout << "Prepared " << rows.size() << " dense/native tensors, "
            << erows.size() << " expert layers, " << at << " dense bytes; "
            << converted << " BF16 conversions\n";
}
int main(int argc, char **argv) {
  if (argc != 3) {
    std::cerr << "usage: maya-pack FIRST.gguf OUTPUT_DIRECTORY\n";
    return 2;
  }
  try {
    run(argv[1], argv[2]);
    return 0;
  } catch (const std::exception &e) {
    std::cerr << "maya-pack: " << e.what() << '\n';
    return 1;
  }
}
