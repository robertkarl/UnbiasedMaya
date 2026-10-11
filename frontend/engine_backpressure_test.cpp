#include "engine.hpp"
#include <cassert>
#include <condition_variable>
#include <deque>
#include <stdexcept>
#include <thread>

// Enough output to fill a Linux pipe, with the same independent stdin reader
// and per-token STOP check as the backend. This requires no model or CUDA.
int fake_backend() {
    std::setvbuf(stdout, nullptr, _IONBF, 0);
    std::mutex mutex;
    std::condition_variable ready;
    std::deque<std::string> commands;
    std::atomic<bool> stop{false};
    auto reader = std::thread([&] {
        std::string input;
        char byte;
        while (::read(0, &byte, 1) == 1) {
            if (byte != '\n') { input += byte; continue; }
            if (input == "STOP") { stop.store(true); input.clear(); continue; }
            const bool quit = input == "QUIT";
            { std::lock_guard<std::mutex> lock(mutex); commands.push_back(input); }
            input.clear(); ready.notify_one();
            if (quit) return;
        }
    });
    std::puts("READY");
    for (;;) {
        std::unique_lock<std::mutex> lock(mutex);
        ready.wait(lock, [&] { return !commands.empty(); });
        auto command = commands.front(); commands.pop_front(); lock.unlock();
        if (command == "QUIT") break;
        stop.store(false);
        const int token = command.find("99") != std::string::npos ? 99 : 11;
        int count = 0;
        while (count < 100000) {
            std::printf("T %d\n", token); ++count;
            if (stop.load()) break;
        }
        std::printf("DONE %d 1 1 1 stop 0 0\n", count);
    }
    reader.join();
    return 0;
}

int main(int argc, char** argv) {
    if (argc > 1) return fake_backend();
    strata_frontend::Engine engine;
    std::string error;
    assert(engine.start({argv[0], "--fake-backend"}, error, argv[0]));
    assert(engine.wait_ready(error));
    auto failed = engine.generate("GEN 100000 ", {11}, [](int32_t) -> bool {
        throw std::runtime_error("partial tool JSON formatter failure");
    });
    assert(!failed.ok);
    int count = 0;
    auto next = engine.generate("GEN 100000 ", {99}, [&](int32_t token) {
        assert(token == 99); ++count; return true;
    });
    assert(next.ok && count == 100000);
    auto cancel = engine.generate("GEN 100000 ", {11}, [](int32_t) { return false; });
    assert(cancel.ok);
    count = 0;
    next = engine.generate("GEN 100000 ", {99}, [&](int32_t token) {
        assert(token == 99); ++count; return true;
    });
    assert(next.ok && count == 100000);
    engine.quit();
    std::puts("PASS: large responses cannot fill an abandoned pipe or leak into the next request");
}
