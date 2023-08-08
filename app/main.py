import uvicorn
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware


from load_model import Llama2
from openai_api_schemas import (
    ChatCompletionChoice,
    ChatCompletionRequestBody,
    ChatCompletionResponse,
    ChatResponseMessage,
    Usage
)
from llama2_prompt import extract_answer, construct_llama2_prompt


app = FastAPI(title='Inference API for Llama 2')
app.add_middleware(CORSMiddleware,
                   allow_origins=['*'],
                   allow_methods=['*'],
                   allow_headers=['*']
                   )


@app.post('/v1/chat/completions', response_model=ChatCompletionResponse)
async def chat_completions(inputs: ChatCompletionRequestBody):
    llm = Llama2()

    prompt = construct_llama2_prompt(inputs.messages)
    print(prompt)
    outputs = llm(prompt)

    msg = ChatResponseMessage(
        content=extract_answer(outputs[0]['generated_text'])
    )
    chat_response = ChatCompletionResponse(
        choices=[ChatCompletionChoice(message=msg)],
        model=inputs.model,
        usage=Usage())

    return chat_response

if __name__ == '__main__':
    Llama2()
    uvicorn.run(app, host='0.0.0.0', port=3000)
