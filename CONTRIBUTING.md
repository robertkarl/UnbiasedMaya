# Working on UnbiasedMaya

The supported build, preparation, server, and tests must not invoke Python or
fetch dependencies implicitly. Prepare pinned dependencies explicitly on a
connected staging machine; validate the exported source on an offline machine.

Keep inference math changes separate from frontend and packaging changes.
Update UPSTREAM.md when changing pinned source or local dependency patches.
Run `make test` and `make test-chat VOCAB=...` after relevant changes. Record
actual-model API, function-call, and performance results for runtime changes.

Never report a model deployment as complete based only on compilation or a CLI
exit code. Check memory, model output, tool arguments, and any agent artifact.
Preserve existing inference services and unrelated benchmark runs.
