#include "engine.hpp"
#include <cassert>
#include <stdexcept>

int main(int argc, char** argv) {
    if (argc > 1 && std::string(argv[1]) == "--fake-engine") {
        std::puts("READY"); std::fflush(stdout);
        char* line = nullptr; size_t capacity = 0;
        while (::getline(&line, &capacity, stdin) > 0) {
            const std::string request(line);
            if (request == "QUIT\n") break;
            if (request == "STOP\n") continue;
            if (request.rfind("GEN ", 0) != 0) continue;
            if (request.find("88") != std::string::npos) {
                std::printf("T 88\nDONE 1 2 1 1 stop 0 0\n");
                std::fflush(stdout);
                continue;
            }
            const int base = request.find("99") != std::string::npos ? 99 : 11;
            std::printf("T %d\nT %d\nDONE 2 1 1 1 stop 0 0 1 7 9\n", base, base+1);
            std::fflush(stdout);
        }
        std::free(line);
        return 0;
    }
    strata_frontend::Engine engine;
    std::string error;
    assert(engine.start({argv[0], "--fake-engine"}, error, argv[0]));
    assert(engine.wait_ready(error));
    auto failed = engine.generate("GEN 2 ", {11}, [](int32_t) -> bool {
        throw std::runtime_error("synthetic partial tool JSON diff error");
    });
    assert(!failed.ok && failed.error.find("synthetic") != std::string::npos);
    std::vector<int32_t> tokens;
    auto next = engine.generate("GEN 2 ", {99}, [&](int32_t token) { tokens.push_back(token); return true; });
    assert(next.ok && next.reused_n == 1 && (tokens == std::vector<int32_t>{99,100}));
    auto cancelled = engine.generate("GEN 2 ", {11}, [](int32_t) { return false; });
    assert(cancelled.ok);
    tokens.clear();
    next = engine.generate("GEN 2 ", {99}, [&](int32_t token) { tokens.push_back(token); return true; });
    assert(next.ok && (tokens == std::vector<int32_t>{99,100}));
    auto mismatched = engine.generate("GEN 2 ", {88}, [](int32_t) { return true; });
    assert(!mismatched.ok && !engine.alive());
    assert(mismatched.error.find("prompt size mismatch") != std::string::npos);
    auto rejected = engine.generate("GEN 2 ", {99}, [](int32_t) { return true; });
    assert(!rejected.ok && rejected.produced == 0);
    engine.quit();
    std::puts("PASS: formatter exceptions and client cancellation drain the old request before the next request");
}
