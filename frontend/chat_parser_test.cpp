// CPU-only tests using Maya-S's real vocabulary/template, without inference.
#define main unused_serving_main
#include "main.cpp"
#undef main
#include <iostream>
int strata_engine_main(int, char **) { return 99; }
static void check(bool ok, const std::string &why) {
  if (!ok)
    throw std::runtime_error(why);
}
int main(int argc, char **argv) {
  if (argc != 2)
    return 2;
  try {
    llama_backend_init();
    Server server;
    server.opt.ctx = 8192;
    auto params = llama_model_default_params();
    params.vocab_only = true;
    server.model = llama_model_load_from_file(argv[1], params);
    check(server.model, "vocabulary load failed");
    server.vocab = llama_model_get_vocab(server.model);
    server.tmpls = maya_templates(server.model);
    auto request = common_json::parse_no_throw(
        R"({"messages":[{"role":"user","content":"Compute 17 + 25 using add."}],"reasoning_effort":"none","temperature":0,"tools":[{"type":"function","function":{"name":"add","parameters":{"type":"object","properties":{"a":{"type":"integer"},"b":{"type":"integer"}},"required":["a","b"]}}}]})");
    Prepared p;
    std::string error;
    check(server.prepare(request, p, error), "prepare: " + error);
    check(p.header.find("temperature=0") != std::string::npos,
          "greedy sampling missing");

    auto msg = common_chat_parse(
        "<tool_call>add<arg_key>b</arg_key><arg_value>25</"
        "arg_value><arg_key>a</arg_key><arg_value>17</arg_value></tool_call>",
        false, p.parser);
    check(msg.tool_calls.size() == 1, "GLM tool call not parsed");
    auto args = common_json::parse_no_throw(msg.tool_calls[0].arguments);
    check(args.at("a").get<int>() == 17 && args.at("b").get<int>() == 25,
          "tool argument order/types lost");
    request["tools"][0]["function"]["parameters"]["properties"]["note"] =
        common_json::parse_no_throw(R"({"type":"string"})");
    check(server.prepare(request, p, error), "optional argument prepare");
    auto optional = common_chat_parse(
        "<tool_call>add<arg_key>note</arg_key><arg_value>line 1\nline "
        "2</arg_value>"
        "<arg_key>b</arg_key><arg_value>25</arg_value><arg_key>a</"
        "arg_key><arg_value>17</arg_value></tool_call>",
        false, p.parser);
    check(optional.tool_calls.size() == 1,
          "optional-before-required call lost");
    auto optional_args =
        common_json::parse_no_throw(optional.tool_calls[0].arguments);
    check(optional_args.at("note").get<std::string>() == "line 1\nline 2",
          "string argument corrupted");
    bool refused = false;
    try {
      auto missing =
          common_chat_parse("<tool_call>add<arg_key>a</arg_key><arg_value>17</"
                            "arg_value></tool_call>",
                            false, p.parser);
      refused = missing.tool_calls.empty();
    } catch (const std::exception &) {
      refused = true;
    }
    check(refused, "missing required argument accepted");
    request["messages"].push_back(common_json::parse_no_throw(
        R"({"role":"assistant","content":null,"tool_calls":[{"id":"call_1","type":"function","function":{"name":"add","arguments":"{\"a\":17,\"b\":25}"}}]})"));
    request["messages"].push_back(common_json::parse_no_throw(
        R"({"role":"tool","tool_call_id":"call_1","content":"42"})"));
    check(server.prepare(request, p, error), "tool result template: " + error);
    auto final = common_chat_parse("The answer is 42.", false, p.parser);
    check(final.content.find("42") != std::string::npos, "plain answer lost");
    request["reasoning_effort"] = "medium";
    check(server.prepare(request, p, error), "thinking prepare: " + error);
    auto thought = common_chat_parse(
        "Let me calculate.</think>The answer is 42.", false, p.parser);
    check(thought.content.find("42") != std::string::npos &&
              !thought.reasoning_content.empty(),
          "reasoning split failed");
    for (const std::string text :
         {"1234567890", "你好，世界", "emoji 🦙 and newlines\n\tspaces"}) {
      auto ids = common_tokenize(server.vocab, text, false, true);
      check(common_detokenize(server.vocab, ids, true) == text,
            "tokenizer round trip failed");
    }
    request["presence_penalty"] = 0.5;
    check(!server.prepare(request, p, error), "unsupported penalty accepted");
    llama_model_free(server.model);
    std::cout << "GLM tokenizer, template, reasoning, and tool tests passed\n";
    return 0;
  } catch (const std::exception &e) {
    std::cerr << e.what() << '\n';
    return 1;
  }
}
