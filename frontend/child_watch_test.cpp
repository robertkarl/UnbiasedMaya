#include "child_watch.hpp"
#include <cassert>
#include <csignal>
#include <cstdio>
#include <sys/resource.h>
#include <unistd.h>

int main() {
    for (bool crash : {false, true}) {
        pid_t pid = ::fork();
        assert(pid >= 0);
        if (pid == 0) {
            const struct rlimit no_core{0, 0};
            ::setrlimit(RLIMIT_CORE, &no_core);
            ::usleep(30000);
            if (crash) ::raise(SIGABRT);
            ::_exit(17);
        }
        std::string why;
        assert(strata_frontend::child_running(pid, why));
        bool exited = false;
        for (int i = 0; i < 100; ++i) {
            if (!strata_frontend::child_running(pid, why)) { exited = true; break; }
            ::usleep(10000);
        }
        assert(exited && pid == -1);
        assert(why == (crash ? "engine killed by signal " + std::to_string(SIGABRT) : "engine exited with code 17"));
        std::puts(why.c_str());
    }
    std::puts("PASS: normal exit and watchdog-style abort reaped and diagnosed");
}
