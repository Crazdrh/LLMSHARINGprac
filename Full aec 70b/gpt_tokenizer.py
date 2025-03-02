"""
gpt_tokenizer.py

Imports DeepSeekTokenizer from the installed 'deepseek_tokenizer' library
and creates a global ds_token instance for encoding text.
"""

from deepseek_tokenizer import ds_token

# Global instance
ds_token = ds_token
"""
Usage:
  from gpt_tokenizer import ds_token
  token_ids = ds_token.encode("Hello world!")
  print(token_ids)
"""
