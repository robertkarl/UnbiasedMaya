#pragma once

#include <cerrno>
#include <cstring>
#include <string>
#include <sys/wait.h>

namespace strata_frontend {
// Called only by the supervisor, which is joined before Engine::quit().
// Reap dead children: kill(pid, 0) would report a zombie as still alive.
inline bool child_running(pid_t& pid, std::string& why) {
    int status = 0;
    const pid_t result = ::waitpid(pid, &status, WNOHANG);
    if (result == 0 || (result < 0 && errno == EINTR)) return true;
    if (result < 0) why = std::string("waitpid: ") + std::strerror(errno);
    else if (WIFSIGNALED(status)) why = "engine killed by signal " + std::to_string(WTERMSIG(status));
    else if (WIFEXITED(status)) why = "engine exited with code " + std::to_string(WEXITSTATUS(status));
    else why = "engine exited unexpectedly";
    pid = -1;
    return false;
}
}
