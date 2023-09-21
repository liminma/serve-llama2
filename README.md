# Serve Llama 2
Implementation of endpoint `/v1/chat/completions` using `FastAPI`:

![](assets/screenshot1.png)

![](assets/screenshot2.png)

## To run:
1. configure API keys in the `app/api_keys.py` file, e.g.,
```
API_KEYS = [
    # enter a list of valid API keys, e.g.,
    '992239417f1e4c0bb768dc204df31289',
]
```
2. start the server,
```bash
python app/main.py
```
Now the server listens at port 3000.

## To call:
It can be called by `openai.ChatCompletion.create()` function,

```bash
>>> import openai
>>> openai.api_base = "http://localhost:3000/v1"
>>> openai.api_key = "992239417f1e4c0bb768dc204df31289"
>>> response = openai.ChatCompletion.create(
...     model="dummy name",
...     messages=[
...         {"role": "user", "content": "how are you?"}
...     ]
... )
>>> print(response.choices[0].message.content)
I'm doing well, thank you for asking! I'm here to assist you with any questions or concerns you may have. Is there anything specific you'd like to know or discuss?
>>>
```

or call the API using `curl` command,

```bash
$ curl -X 'POST' \
  'http://localhost:3000/v1/chat/completions' \
  -H 'accept: application/json' \
  -H 'Authorization: Bearer 992239417f1e4c0bb768dc204df31289' \
  -H 'Content-Type: application/json' \
  -d '{
  "model": "dummy name",
  "messages": [
    {
      "role": "user",
      "content": "Hello, how are you?"
    }
  ]
}'
{"choices":[{"index":0,"message":{"role":"assistant","content":"Hello! I'm doing well, thanks for asking. I'm here to assist you with any questions or concerns you may have. Is there anything specific you'd like to know or discuss?","name":null,"function_call":null},"finish_reason":null}],"created":1695259828,"id":"chatcmpl-145dea8910a142b7b6767330f776e22a","model":"dummy name","object":"chat.completion","usage":{"completion_tokens":0,"prompt_tokens":0,"total_tokens":0}}(venvServeLlama2)
```
