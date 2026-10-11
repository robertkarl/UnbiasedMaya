.DEFAULT_GOAL := build
BUILD_DIR ?= build
CUDA_ARCHS ?= 86
JOBS ?= 4
CUDA ?= ON
.PHONY: build pack test
build:
	@test -f third_party/llama.cpp/ggml/CMakeLists.txt || { echo 'Copy pinned dependencies from the connected host first'; exit 1; }
	cmake -S . -B $(BUILD_DIR) -DCMAKE_BUILD_TYPE=Release -DSTRATA_ENABLE_CUDA=$(CUDA) -DSTRATA_NATIVE_EXPERTS=ON -DSTRATA_PORTABLE=ON -DSTRATA_BUILD_TESTS=OFF "-DCMAKE_CUDA_ARCHITECTURES=$(CUDA_ARCHS)"
	cmake --build $(BUILD_DIR) -j $(JOBS) --target maya-pack $(if $(filter ON,$(CUDA)),unbiased-maya)
pack:
	@test -n "$(MODEL)" -a -n "$(PACK)" || { echo 'Set MODEL=first.gguf PACK=output-directory'; exit 1; }
	$(BUILD_DIR)/maya-pack "$(MODEL)" "$(PACK)"
test:
	mkdir -p $(BUILD_DIR)/frontend-tests
	$(CXX) -std=c++20 -Wall -Wextra -Werror frontend/child_watch_test.cpp -o $(BUILD_DIR)/frontend-tests/child-watch-test
	$(BUILD_DIR)/frontend-tests/child-watch-test
	$(CXX) -std=c++20 -Wall -Wextra -Werror frontend/engine_test.cpp -o $(BUILD_DIR)/frontend-tests/engine-test
	$(BUILD_DIR)/frontend-tests/engine-test
	$(CXX) -std=c++20 -Wall -Wextra -Werror -pthread frontend/engine_backpressure_test.cpp -o $(BUILD_DIR)/frontend-tests/engine-backpressure-test
	$(BUILD_DIR)/frontend-tests/engine-backpressure-test
	cmake --build $(BUILD_DIR) -j $(JOBS) --target maya-pack-test
	$(BUILD_DIR)/maya-pack-test
.PHONY: test-chat
test-chat:
	@test -n "$(VOCAB)" || { echo 'Set VOCAB=pack/tokenizer.gguf'; exit 1; }
	cmake --build $(BUILD_DIR) -j $(JOBS) --target maya-chat-test
	$(BUILD_DIR)/maya-chat-test "$(VOCAB)"
