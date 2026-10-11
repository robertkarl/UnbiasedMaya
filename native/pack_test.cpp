#define main maya_pack_program_main
#include "pack.cpp"
#undef main
#include "gguf.h"
static void check(bool ok, const char *message) { require(ok, message); }
static void fixture(const fs::path &path, bool second, bool broken = false) {
  auto *g = gguf_init_empty();
  gguf_set_val_str(g, "general.architecture", "glm5-next");
  gguf_set_val_u32(g, "glm5-next.block_count", 2);
  gguf_set_val_u32(g, "glm5-next.nextn_predict_layers", 1);
  ggml_init_params p{1024 * 1024, nullptr, false};
  auto *ctx = ggml_init(p);
  auto add = [&](const char *name, ggml_type type, int64_t a, int64_t b,
                 int64_t c) {
    auto *t = ggml_new_tensor_3d(ctx, type, a, b, c);
    ggml_set_name(t, name);
    std::memset(t->data, 0, ggml_nbytes(t));
    if (type == GGML_TYPE_F32) {
      auto *f = (float *)t->data;
      f[0] = 1.00390625f;
      f[1] = 1.01171875f; // opposite halfway-even cases
    }
    gguf_add_tensor(g, t);
  };
  if (!second) {
    // Router must be 2-D in the actual GGUF.
    auto *t = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, 256, 2);
    ggml_set_name(t, "blk.0.ffn_gate_inp.weight");
    std::memset(t->data, 0, ggml_nbytes(t));
    ((float *)t->data)[0] = 1.00390625f;
    ((float *)t->data)[1] = 1.01171875f;
    gguf_add_tensor(g, t);
    add("blk.0.ffn_gate_exps.weight", GGML_TYPE_IQ2_XXS, 256, 2, 2);
    add("blk.0.ffn_up_exps.weight", GGML_TYPE_IQ2_XXS, 256, 2, 2);
    add("blk.0.attn_k_b.weight", GGML_TYPE_F32, 2, 3, 4);
    add("blk.1.hc_attn_fn.weight", GGML_TYPE_F32, 2, 1, 1); // MTP excluded
    if (broken)
      add("blk.0.unsupported.weight", GGML_TYPE_Q4_0, 256, 1, 1);
  } else
    add("blk.0.ffn_down_exps.weight", GGML_TYPE_IQ3_XXS, 256, 2, 2);
  check(gguf_write_to_file(g, path.c_str(), false), "write fixture");
  gguf_free(g);
  ggml_free(ctx);
}
int main() {
  fs::path dir = fs::temp_directory_path() /
                 ("maya-pack-test-" + std::to_string(getpid()));
  fs::create_directory(dir);
  try {
    auto first = dir / "test-00001-of-00002.gguf",
         second = dir / "test-00002-of-00002.gguf";
    fixture(first, false);
    bool rejected = false;
    try {
      run(first, dir / "missing");
    } catch (...) {
      rejected = true;
    }
    check(rejected, "missing shard accepted");
    fixture(second, true);
    run(first, dir / "pack");
    std::ifstream dense(dir / "pack/dense.bin", std::ios::binary);
    uint16_t values[2];
    dense.read((char *)values, 4);
    check(values[0] == 0x3f80 && values[1] == 0x3f82,
          "BF16 tie rounding mismatch");
    std::ifstream ix(dir / "pack/index.txt");
    std::string all((std::istreambuf_iterator<char>(ix)), {});
    check(all.find("blk.1.") == std::string::npos, "MTP leaked into trunk");
    check(all.find(" 6 4 0 0 1 ") != std::string::npos,
          "3-D flattening incorrect");
    std::ifstream ex(dir / "pack/native_experts.txt");
    std::string line;
    std::getline(ex, line);
    std::getline(ex, line);
    check(line.find("test-00001-of-00002.gguf test-00001-of-00002.gguf "
                    "test-00002-of-00002.gguf") != std::string::npos,
          "cross-shard expert references wrong");
    strata::GgufFile f(first.string()), s(second.string());
    std::istringstream row(line);
    uint64_t l, gu, d, off, blob, go, uo, dow;
    row >> l >> gu >> d >> off >> blob >> go >> uo >> dow;
    check(go == f.data_start() + f.find("blk.0.ffn_gate_exps.weight")->offset &&
              dow ==
                  s.data_start() + s.find("blk.0.ffn_down_exps.weight")->offset,
          "absolute offsets wrong");
    check(blob == 2 * 2 * 66 + 2 * 98, "expert blob size wrong");
    rejected = false;
    try {
      run(first, dir / "pack");
    } catch (...) {
      rejected = true;
    }
    check(rejected, "overwrote existing pack");
    fixture(first, false, true);
    rejected = false;
    try {
      run(first, dir / "bad");
    } catch (...) {
      rejected = true;
    }
    check(rejected, "unknown quantized dense tensor accepted");
    fs::resize_file(second, fs::file_size(second) - 32);
    rejected = false;
    try {
      run(first, dir / "truncated");
    } catch (...) {
      rejected = true;
    }
    check(rejected, "truncated shard accepted");
    fs::remove_all(dir);
    std::cout << "Pack regression tests passed\n";
    return 0;
  } catch (const std::exception &e) {
    std::cerr << e.what() << "; fixtures retained: " << dir << '\n';
    return 1;
  }
}
