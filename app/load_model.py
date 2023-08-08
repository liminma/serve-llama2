import os
from dotenv import find_dotenv, load_dotenv

import torch
from transformers import pipeline, GenerationConfig


load_dotenv(find_dotenv())
HF_TOKEN = os.environ.get('HF_TOKEN')
MODEL_ID = os.environ.get('MODEL_ID')


class Llama2(object):
    """A singleton
    """
    def __new__(cls):
        """Load the model using transformers pipeline if it hasn't been loaded yet.

        Returns:
            a transformers pipeline
        """
        if not hasattr(cls, 'instance'):
            # load model
            print(f'loading model {MODEL_ID} ...')
            gen_config = GenerationConfig.from_pretrained(
                MODEL_ID, token=HF_TOKEN)
            gen_config.max_new_tokens = 4096
            gen_config.temperature = 0.0

            cls.instance = super().__new__(cls)
            cls.instance = pipeline(
                model=MODEL_ID,
                use_auth_token=HF_TOKEN,
                task="text-generation",
                device_map='auto',
                generation_config=gen_config,
                model_kwargs={
                    'load_in_4bit': True,
                    'bnb_4bit_quant_type': 'nf4',
                    'bnb_4bit_use_double_quant': True,
                    'bnb_4bit_compute_dtype': torch.bfloat16},
            )

        return cls.instance
