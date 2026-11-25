"""Build the model's native reasoning prompt and decode generated token IDs."""


def tokenizer_for(alias, path):
    if alias == "magistral":
        from mistral_common.protocol.instruct.messages import SystemMessage, UserMessage
        from mistral_common.protocol.instruct.request import ChatCompletionRequest
        from mistral_common.tokens.tokenizers.mistral import MistralTokenizer
        from mistral_common.tokens.tokenizers.tekken import SpecialTokenPolicy

        tokenizer = MistralTokenizer.from_file(str(path / "tekken.json"))
        system = (path / "SYSTEM_PROMPT.txt").read_text().strip()

        def encode(text):
            request = ChatCompletionRequest(messages=[SystemMessage(content=system), UserMessage(content=text)])
            return tokenizer.encode_chat_completion(request).tokens

        def decode(ids):
            return tokenizer.instruct_tokenizer.tokenizer.decode(ids, special_token_policy=SpecialTokenPolicy.IGNORE)

        return encode, decode

    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(path)

    def encode(text):
        return tokenizer.apply_chat_template([{"role": "user", "content": text}], tokenize=True,
                                             add_generation_prompt=True, enable_thinking=True,
                                             reasoning_effort="high")

    def decode(ids):
        return tokenizer.decode(ids, skip_special_tokens=True)

    return encode, decode
