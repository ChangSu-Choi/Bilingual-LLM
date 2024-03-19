# LREC-Coling
[DEMO](https://644a58fbc08f02d0be.gradio.live/)

## Model Details
- **Model Developers**: Anonymous
- **Model Architecture** : This Model is an auto-regressive language model that uses an optimized transformer architecture. The tuned versions use pre-training by multilingual text dataset and supervised fine-tuning by multi-task instruction.

## Dataset
- Pretraining Dataset: En-Ko Aligned Pretraining Dataset
- SFT Dataset: Aligning the English LIMA dataset with Korean language style, creating a Korean LIMA dataset that includes Korean cultural context. Utilizing KO,EN LIMA data for Instruction tuning.


## Prompt Template

```
[INST] <<SYS>>\n
You are a helpful assistant. 당신은 유능한 AI 어시스턴트 입니다.\n
<</SYS>>

{instruction} [/INST]
```


## Examples  
```
[INST] <<SYS>>\n
You are a helpful assistant. 당신은 유능한 AI 어시스턴트 입니다.\n
<</SYS>>

여자친구 만드는 방법을 알려줘! [/INST]
```

## Response  
```
[INST] <<SYS>>\n
You are a helpful assistant. 당신은 유능한 AI 어시스턴트 입니다.\n
<</SYS>>

여자친구 만드는 방법을 알려줘! [/INST]
여자 친구를 구하는 것은 어렵고 때로는 좌절감과 낙담으로 가득 차게 만들 수 있습니다. 하지만 그렇지 않더라도 여러분의 노력이 결실을 맺을 수 있다면, 그것은 평생 지속될 수 있는 진정한 사랑에 다가갈 수 있는 기회일 것입니다. 많은 사람들이 "어떻게 하면 여자 친구를 사귀게 될까요?"라고 물어봅니다. 하지만 더 나은 질문은 "어떻게 하면 내가 원하는 사람과 함께 좋은 관계를 맺을 수 있을까요?"입니다.
단계별 계획 세우기
1. 목표를 설정하세요. 목표를 설정하고 달성하기 위해 필요한 시간을 정하세요. 예를 들어, 한 달 또는 두 달 동안 여자 친구를 사귀기를 원한다면, 이 기간 동안 매일 계획을 세울 수 있도록 시간을 정하세요. 목표를 설정하면 여러분이 해야 할 일들을 목록화할 수 있게 되고, 여러분이 해야 할 일이 무엇인지 알 수 있을 뿐만 아니라, 여러분이 해야 할 일을 완료하기 위한 시간도 알 수 있게 됩니다.
2. 현재 상태를 파악하세요. 현재 자신의 상황을 파악해야 합니다. 현재 어떤 식으로 여자 친구를 만나고 있으며, 어떤 스타일의 여자 친구를 만나고 있는지 생각해 보세요. 또한, 여러분이 원하는 여자 친구와 어떻게 다른지 생각해 보세요. 예를 들어, 여러분이 원하는 여자 친구는 운동을 좋아하거나, 담배를 피우거나, 술을 마시는 사람이 아닌지 생각해 보세요. 그런 다음, 여러분이 원하는 여자 친구와 현재 가지고 있는 여자 친구와의 차이점을 적어보세요.
3. 여자 친구를 찾기 위한 방법을 선택하세요. 여자 친구를 찾는 방법을 선택합니다. 예를 들어, 온라인 데이팅 사이트나 앱을 사용하여 여자 친구를 찾을 수도 있고, 친구 소개를 통해 여자 친구를 찾을 수도 있습니다. 또 다른 방법은 지역 클럽이나 단체에서 여자 친구를 찾아보는 것입니다.
4. 여자 친구를 찾기 위한 프로필을 만드세요. 온라인 데이팅 웹사이트나 앱을 사용하여 여자 친구를 찾을 경우, 프로필을 작성하세요. 프로필에는 여러분의 사진과 함께 여러분의 이름, 나이, 관심 분야, 취미 활동, 직업, 외모, 성격, 가족 구성원, 종교적 신념 등을 자세히 설명하세요.
행복한 연애 하세요☺️
```

## Usage
```
from transformers import AutoTokenizer, AutoModelForCausalLM
from transformers import GenerationConfig
import torch

model_name = "Bllossom"

generation_config = GenerationConfig(
    temperature = 0.2,
    top_p = 0.9,
    top_k = 40,
    max_length = 1024,
    repetition_penalty = 1.1,
    do_sample = True
)

device = 'cuda' if torch.cuda.is_available() else 'cpu'
model = AutoModelForCausalLM.from_pretrained(model_name).to(device)
tokenizer = AutoTokenizer.from_pretrained(model_name)

PROMPT = "[INST] <<SYS>>\n {system_prompt}\n <</SYS>>\n\n {instruction} [/INST]"
SYSTEM_PROMPT = "You are a helpful assistant. 당신은 훌륭한 비서입니다."
instruction = """
"Italy covers an area of 301,340 km^2 (116,350 sq mi), with a population of nearly 60 million. it is the tenth-largest country by land area in the European continent and the third-most populous member state of the European Union. Its capital and largest city is Rome."를 한국어로 해석해줘.
"""

tokenized_sample = tokenizer(PROMPT.format(system_prompt = SYSTEM_PROMPT, instruction = instruction), max_length=1024, return_tensors='pt').to(device)
print(tokenized_sample)
outputs = model.generate(**tokenized_sample, generation_config = generation_config)

print(tokenizer.decode(outputs[0]))
```
response:
```
이탈리아는 유럽 대륙의 면적 301,3041 제곱킬로미터(116,30050제곱피 미터) 내에 위치한 국가이며 인구가 거의 60만 명으로 세계 10대 나라 중 하나입니다. 그리고 유럽 연합에서 가장 큰 회원국 중 하나입니다. 로마는 이탈리아의 수도이자 가장 큰 도시입니다.
```

## Hardware and Software  
NVIDIA RTX A6000  
- nvidia driver : 530.30.02
- CUDA version : 11.7


## Evaluation results by GPT4
- Korean language proficiency
<figure>
  <img src="./Kor Eval.png" width="500" >
</figure>

- English language proficiency
<figure>
  <img src="./Eng Eval.png" width="500" >
</figure>

## Q/A
Q: 누전차단기가 내려가는 이유는 무엇입니까?
___
A: 누전차단기는 자동적으로 회로에 이상이 있을 때 차단하기 위한 장치입니다. 😀
전기회로에서 이상이 발생하면 회로의 전압과 전류가 허용되는 범위를 초과하여 증가하게 되며, 이는 회로에 손상을 입힐 수 있기 때문입니다. 이러한 상황에서는 누전차단기의 트리거 지점(trip point)을 설정하는 것이 필요합니다. 트리거 지점은 누전차단기의 동작을 제어하는 중요한 요소이며, 이를 설정하는 것은 매우 중요하며 신중하게 고려되어야 합니다. 누전차단기의 트리거 지점은 다음과 같은 요인에 따라 결정됩니다:

회로에 흐르는 전류와 전력량이 어느 정도인지
회로에 사용되는 기기나 장치의 종류 및 특성
회로에 있는 기기나 장치의 개수
회로에 흐르는 전압의 변동폭과 주파수
회로에 영향을 미치는 외부 요인(예: 정전 등)
누전차단기는 또한 리트리그 기능(retriggerable) 또는 일회용(non-retriggerable) 기능을 가지고 있습니다. 리트리그 기능이란 누전차단기가 다시 트리거되지 않는 시간인 지연시간(delay time) 동안에는 다시 트리거되지 않는다는 의미입니다. 예를 들어, 누전이 감지되고 즉시 차단되더라도 다시 누전이 감지될 때까지 기다렸다가 차단할 수도 있습니다. 반면에 일회성 기능은 누전이 감지되면 차단하고 이후에는 다시 트리거되지 않습니다. 누전이 감지된 후 일정 시간이 지나면 차단기를 재시동해야 하는 경우에만 일회성 기능이 적용됩니다. 누전차단기는 일반적으로 리트리그 기능이 더 신뢰성이 높지만, 일회성 기능도 상황에 따라 유용할 수 있습니다. 누전차단기는 또한 차단할 때 어떤 동작을 취할 것인지를 결정하는 또 다른 중요한 설정이 있습니다. 누전차단기가 차단되면 회로에 남아있는 잔류전류를 제거하거나 회로에 남아있는 전압을 제거하는 두 가지 방법이 있으며, 각 방법마다 장단점이 있습니다. 누전차단기에 대한 자세한 내용은 아래 링크를 참조하세요! ^^:

https://en.wikipedia.org/wiki/Circuit_breaker#Overcurrent
항상 안전을 최우선으로 생각하시고, 전기적 위험으로부터 보호받으세요!
___
Q: 협력공인중개사란 무엇인가요?
___
A: 부동산 중개업에서 자주 볼 수 있는 직업으로, 일반 공인중개사와 함께 부동산 거래를 도와주고 대리수수료를 받는 사람을 의미합니다. 이들은 일반적으로 단독 또는 팀을 이루어 활동하며, 일부 지역에서는 이러한 '부자' 중개인에게 특별히 '공인'되지 않은 다른 '비공인' 중개인이 존재하기도 합니다. 이들의 역할과 권한에 대한 자세한 내용은 지역법령 및 해당 지역 부동산협회(또는 유사 명칭)에서 제공하는 지침서 등을 참조하시기 바랍니다.
___
Q: 파스타의 주재료가 뭐야?
___
A: 파스타는 다양한 재료로 맛있게 즐길 수 있는 음식입니다.😋 일반적으로 기본 재료는 다음과 같습니다:

면류: 스파게티, 라비올리, 파스타 등 다양한 종류가 있습니다.
소스: 토마토, 올리브 오일, 양파, 가지, 마늘, 페페로니, 알렉산더 등 선택할 수 있습니다.
이 외에도 고기, 생선, 채소 등을 추가하여 자신만의 특별한 파스타 요리를 만들 수도 있답니다. :)
___
Q: "Italy covers an area of 301,340 km^2 (116,350 sq mi), with a population of nearly 60 million. it is the tenth-largest country by land area in the European continent and the third-most populous member state of the European Union. Its capital and largest city is Rome."를 한국어로 해석해줘.
___
A: 이탈리아의 면적은 301,3040km^2(이하 166,350평방마일}이며, 인구는 거의 60만 명이며. 이는 유럽 대륙에서 10번째로 큰 나라이자 유럽 연합 (EU)의 회원국 중 3번째로 인구가 많은 국가입니다. 로마는 이탈리아의 수도이자 가장 큰 도시입니다.
___
Q: Translate this sentence in English

지구(地球, 영어: Earth)는 태양으로부터 세 번째 행성이며, 조금 두꺼운 대기층으로 둘러싸여 있고, 지금까지 발견된 지구형 행성 가운데 가장 크다. 지구는 45억 6700만 년 전 형성되었으며, 용암 활동이 활발했던 지구와 행성 테이아의 격렬한 충돌로 생성되었을 달을 위성으로 둔다. 지구의 중력은 우주의 다른 물체, 특히 태양과 지구의 유일한 자연위성인 달과 상호작용한다. 지구와 달 사이의 중력 작용으로 조석 현상이 발생한다.
___
A: The earth (地球, 영어:Earth) is the third planet from the sun, and it is the largest of all the known Earth-like planets. The earth was formed about 45.6 billion years ago by the violent collision of the early terrestrial planet and a large moon that generated volcanic activity on the young earth. The force of gravity on the earth is influenced by other celestial bodies, particularly the sun and the earth's only natural satellite, the moon. This force also produces the tide phenomenon.
