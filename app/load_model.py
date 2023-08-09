import os
from dotenv import find_dotenv, load_dotenv

import torch
from transformers import (pipeline, GenerationConfig,
                          LlamaForCausalLM, LlamaTokenizer, BitsAndBytesConfig)


load_dotenv(find_dotenv())
HF_TOKEN = os.environ.get('HF_TOKEN')
MODEL_ID = os.environ.get('MODEL_ID')
TEMPERATURE = float(os.environ.get('TEMPERATURE'))
MAX_NEW_TOKENS = int(os.environ.get('MAX_NEW_TOKENS'))


class LLM(object):
    """A singleton
    """
    def __new__(cls):
        """Load the model if it hasn't been loaded yet.

        Returns:
            a transformers pipeline
        """
        if not hasattr(cls, 'instance'):
            cls.instance = super().__new__(cls)
            cls.instance = cls._load_model()

        return cls.instance

    @staticmethod
    def _load_model():
        print(f'loading model {MODEL_ID} ...')

        # load model using bitsandbytes lib.
        bitsandbytes_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type='nf4',
            bnb_4bit_use_double_quant=True,
            bnb_4bit_compute_dtype=torch.bfloat16
        )

        model = LlamaForCausalLM.from_pretrained(
            MODEL_ID,
            quantization_config=bitsandbytes_config,
            device_map="auto",
            token=HF_TOKEN,
        )

        tokenizer = LlamaTokenizer.from_pretrained(MODEL_ID, token=HF_TOKEN)

        gen_config = GenerationConfig.from_pretrained(MODEL_ID, token=HF_TOKEN)
        gen_config.temperature = TEMPERATURE
        gen_config.max_new_tokens = MAX_NEW_TOKENS
        gen_config.repetition_penalty = 1.1

        print('>>>>>>>>>>>>', gen_config)

        return pipeline(
            task="text-generation",
            model=model,
            tokenizer=tokenizer,
            device_map='auto',
            generation_config=gen_config,
        )
