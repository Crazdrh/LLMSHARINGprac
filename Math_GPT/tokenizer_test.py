from tokenizer_file import MathTokenizer

def test_tokenizer():
    # 1) Instantiate the tokenizer
    tokenizer = MathTokenizer()

    # 2) Prepare sample expressions
    expressions = input("Enter Your Math Problem: ")

    # 3) Tokenize each expression independently
    tokenized_expressions = [tokenizer.tokenize(expr) for expr in expressions]

    print("---- Tokenized Results ----")
    for expr, tokens in zip(expressions, tokenized_expressions):
        print(f"Expression: {expr}\nTokens: {tokens}\n")

    # 4) Build the vocabulary
    tokenizer.build_vocab(tokenized_expressions)
    print("Vocabulary size after build:", len(tokenizer))

    # 5) Encode and decode each expression
    print("---- Encoding & Decoding ----")
    for expr in expressions:
        encoded = tokenizer.encode(expr)  # includes [SOS] and [EOS] by default
        decoded = tokenizer.decode(encoded, skip_special_tokens=True)
        print(f"Original : {expr}")
        print(f"Encoded  : {encoded}")

if __name__ == "__main__":
    test_tokenizer()
