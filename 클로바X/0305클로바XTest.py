# -*- coding: utf-8 -*-

import getpass
import os
import json
import requests

class CompletionExecutor:
    def __init__(self, host, api_key, request_id):
        self._host = host
        self._api_key = api_key
        self._request_id = request_id

    def execute(self, completion_request):
        headers = {
            'Authorization': f'Bearer {self._api_key}',
            'X-NCP-CLOVASTUDIO-REQUEST-ID': self._request_id,
            'Content-Type': 'application/json; charset=utf-8',
            'Accept': 'text/event-stream'
        }

        response = requests.post(
            f'{self._host}/testapp/v1/chat-completions/HCX-003',
            headers=headers,
            json=completion_request,
            stream=True
        )

        for line in response.iter_lines():
            if line:
                print(line.decode("utf-8"))

if __name__ == '__main__':
    completion_executor = CompletionExecutor(
        host='https://clovastudio.stream.ntruss.com',
        api_key='nv-5bc7bf78b1e04bd9a5cc52719542d6ee58dn',
        request_id='944b72ec389046ee8a80547ee10be77e'
    )

    preset_text = [{"role":"system","content":"sdfsd"},{"role":"user","content":"넌 뭐하냐\n\n"},{"role":"assistant","content":"저는 사용자님을 위해 항상 준비되어 있습니다. 무엇을 도와드릴까요?"},{"role":"user","content":"돈 나가냐? 안 나갈리가 없지"},{"role":"assistant","content":"저와 소통하시는 데에는 별도의 비용이 들지 않습니다. 저는 사용자님께 도움이 되기 위해 존재하며, 언제든지 필요한 정보를 제공해드릴 수 있습니다. \n\n다른 질문이나 요청이 있으시면 언제든지 말씀해주세요."}]

    request_data = {
        'messages': preset_text,
        'topP': 0.8,
        'topK': 0,
        'maxTokens': 256,
        'temperature': 0.5,
        'repeatPenalty': 5.0,
        'stopBefore': [],
        'includeAiFilters': True,
        'seed': 0
    }

    print(preset_text)
    completion_executor.execute(request_data)
