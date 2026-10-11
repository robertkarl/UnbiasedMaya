#include "vocab.hpp"
#include <iostream>
int main(int argc, char **argv) {
  if (argc != 3)
    return 2;
  try {
    write_vocab(argv[1], argv[2]);
    return 0;
  } catch (const std::exception &e) {
    std::cerr << e.what() << '\n';
    return 1;
  }
}
