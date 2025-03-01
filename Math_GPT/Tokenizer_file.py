import re

class MathTokenizer:
    """
    A simple tokenizer for math expressions:
      - Splits on whitespace and recognized math operators (like +, -, /, *, ^, (, ), =).
      - Maintains a vocabulary mapping token -> ID and ID -> token.
      - Allows encoding/decoding of sequences, with optional special tokens.
    """

    def __init__(self, special_tokens=None):
        """
        Constructor to initialize:
          - Special tokens: [PAD], [UNK], [SOS], [EOS].
          - token_to_id: dictionary mapping tokens to integer IDs.
          - id_to_token: dictionary mapping integer IDs back to tokens.
        """
        if special_tokens is None:
            special_tokens = ["[PAD]", "[UNK]", "[SOS]", "[EOS]"]

        self.special_tokens = special_tokens

        # Dictionaries for token <-> ID
        self.token_to_id = {}
        self.id_to_token = {}

        # Initialize special tokens in the vocab
        for i, token in enumerate(self.special_tokens):
            self.token_to_id[token] = i
            self.id_to_token[i] = token

    def tokenize(self, text):
        """
        Splits a raw math expression string into a list of tokens.
        Here we:
          1) Insert spaces around math symbols/operators.
          2) Split on whitespace.
        Example: (a + b)^2 -> ['(', 'a', '+', 'b', ')', '^', '2']
        """
        # Insert spaces around symbols (+, -, /, *, ^, (, ), =, etc.)
        text = re.sub(r"([\+\-\*/\^\(\)=])", r" \1 ", text)
        # Trim and split on whitespace
        tokens = text.strip().split()
        return tokens

    def build_vocab(self, list_of_token_lists):
        """
        Build the vocabulary from a list of tokenized texts.
        list_of_token_lists is something like:
          [['x', '+', '2', '=', '5'], ['sin(x)', '+', 'cos(x)', '=', '1'], ...]
        """
        for token_list in list_of_token_lists:
            for token in token_list:
                # If not already in vocab, assign a new ID
                if token not in self.token_to_id:
                    new_id = len(self.token_to_id)
                    self.token_to_id[token] = new_id
                    self.id_to_token[new_id] = token

    def encode(self, text, add_special_tokens=True):
        """
        Convert a raw text into a list of token IDs.
        Optionally prepends [SOS] and appends [EOS].
        """
        tokens = self.tokenize(text)
        token_ids = []

        # Optionally add [SOS] at the start
        if add_special_tokens:
            sos_id = self.token_to_id["[SOS]"]
            token_ids.append(sos_id)

        # For each token, map it to its ID, or to [UNK] if missing
        for token in tokens:
            if token in self.token_to_id:
                token_ids.append(self.token_to_id[token])
            else:
                unk_id = self.token_to_id["[UNK]"]
                token_ids.append(unk_id)

        # Optionally add [EOS] at the end
        if add_special_tokens:
            eos_id = self.token_to_id["[EOS]"]
            token_ids.append(eos_id)

        return token_ids

    def decode(self, token_ids, skip_special_tokens=True):
        """
        Convert a sequence of token IDs back to a string.
        If skip_special_tokens=True, then [PAD], [UNK], [SOS], [EOS] are ignored.
        """
        tokens = []
        for tid in token_ids:
            token = self.id_to_token.get(tid, "[UNK]")
            if skip_special_tokens and token in self.special_tokens:
                continue
            tokens.append(token)
        return " ".join(tokens)

    def __len__(self):
        """
        Returns the size of the vocabulary (special tokens + built tokens).
        """
        return len(self.token_to_id)
