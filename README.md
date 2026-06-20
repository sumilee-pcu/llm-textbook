# LLM 애플리케이션 입문 — RAG에서 멀티에이전트까지

「LLM 애플리케이션 입문」(이수미 지음)의 **공식 실습 저장소**입니다. 책 본문은 개념이 왜 그렇게 동작하는지를 설명하고, 이 저장소는 그것을 코드로 어떻게 구현하는지를 장별로 담았습니다.

모든 실습은 **구글 코랩에서 무료로, GPU 없이** 따라올 수 있습니다.

## 시작하기 — 코랩과 API 키

3장부터는 Gemini API를 씁니다. 구글 AI 스튜디오(aistudio.google.com)에서 API 키를 무료로 발급받은 뒤, **코랩 보안 비밀(시크릿)**에 등록합니다.

1. 코랩 노트북 왼쪽의 열쇠 아이콘(보안 비밀)을 클릭합니다.
2. 새 보안 비밀을 추가하고 이름을 `GOOGLE_API_KEY`로, 값에 발급받은 키를 넣습니다.
3. 노트북 액세스를 켭니다.
4. 코드에서는 다음처럼 불러옵니다.

```python
from google.colab import userdata
from google import genai
client = genai.Client(api_key=userdata.get("GOOGLE_API_KEY"))
```

10장 웹 검색 실습은 Tavily 키(`TAVILY_API_KEY`)를 같은 방식으로 등록해 사용합니다. 키는 코드에 직접 적지 말고 항상 보안 비밀로 보관합니다.

## Tier 안내

각 실습에는 난이도 단계를 표시했습니다.

- `[T1]` 코랩에서 API 키만 넣으면 바로 실행되는 기본 실습 (대부분이 여기에 해당)
- `[T2]` GPU나 조금 더 긴 실행 시간이 필요한 확장 실습
- `[T3]` 코드를 실행하지 않아도 개념과 설계 판단을 익히는 이론 실습

## 장별 노트북

| 장 | 주제 | 노트북 |
|---|---|---|
| **제1부 토대 — LLM과 프롬프트** | | |
| 1장 | 거대언어모델의 이해 | [열기](https://colab.research.google.com/github/sumilee-pcu/llm-textbook/blob/main/chapters/part1_foundation/ch01_llm_foundations.ipynb) |
| 2장 | 프롬프트 엔지니어링 | [열기](https://colab.research.google.com/github/sumilee-pcu/llm-textbook/blob/main/chapters/part1_foundation/ch02_prompt_engineering.ipynb) |
| 3장 | LLM API 프로그래밍 | [열기](https://colab.research.google.com/github/sumilee-pcu/llm-textbook/blob/main/chapters/part1_foundation/ch03_llm_api.ipynb) |
| **제2부 RAG — 검색으로 지식을 더하다** | | |
| 4장 | 임베딩과 벡터 검색 | [열기](https://colab.research.google.com/github/sumilee-pcu/llm-textbook/blob/main/chapters/part2_rag/ch04_embeddings.ipynb) |
| 5장 | RAG: 검색 증강 생성 | [열기](https://colab.research.google.com/github/sumilee-pcu/llm-textbook/blob/main/chapters/part2_rag/ch05_rag.ipynb) |
| 6장 | RAG 심화와 평가 | [열기](https://colab.research.google.com/github/sumilee-pcu/llm-textbook/blob/main/chapters/part2_rag/ch06_rag_advanced.ipynb) |
| 7장 | GraphRAG: 관계를 묻다 | [열기](https://colab.research.google.com/github/sumilee-pcu/llm-textbook/blob/main/chapters/part2_rag/ch07_graphrag.ipynb) |
| **제3부 에이전트 — 스스로 일하는 LLM** | | |
| 8장 | AI 에이전트의 원리 | [열기](https://colab.research.google.com/github/sumilee-pcu/llm-textbook/blob/main/chapters/part3_agents/ch08_agent_basics.ipynb) |
| 9장 | 에이전트 워크플로: LangGraph | [열기](https://colab.research.google.com/github/sumilee-pcu/llm-textbook/blob/main/chapters/part3_agents/ch09_langgraph.ipynb) |
| 10장 | 멀티 에이전트와 도구 연결 | [열기](https://colab.research.google.com/github/sumilee-pcu/llm-textbook/blob/main/chapters/part3_agents/ch10_multiagent.ipynb) |
| **제4부 심화·운영·종합** | | |
| 11장 | 모델 맞춤: 파인튜닝 맛보기 | [열기](https://colab.research.google.com/github/sumilee-pcu/llm-textbook/blob/main/chapters/part4_advanced_ops/ch11_finetuning.ipynb) |
| 12장 | 운영과 지식 자산화 | [열기](https://colab.research.google.com/github/sumilee-pcu/llm-textbook/blob/main/chapters/part4_advanced_ops/ch12_ops.ipynb) |
| 13장 | 종합 프로젝트: 지식을 가진 에이전트 | [열기](https://colab.research.google.com/github/sumilee-pcu/llm-textbook/blob/main/chapters/part4_advanced_ops/ch13_capstone.ipynb) |

## 실습 환경

| 항목 | 내용 |
|---|---|
| 실행 | 구글 코랩 (CPU, 무료) |
| 언어 | Python 3 |
| 모델 | Gemini API (무료 한도 내 실습 가능) |
| 주요 라이브러리 | google-genai, langchain, langgraph, faiss-cpu, networkx |

각 노트북 첫 셀에서 필요한 라이브러리를 설치하고 한글 폰트를 설정합니다. 모델 이름은 노트북 상단에 명명 상수(`GEMINI_FLASH`)로 두었으니, 모델이 갱신되면 그 한 줄만 바꾸면 됩니다.

## 활용 방법

1. 위 표에서 학습 중인 장의 노트북을 코랩으로 엽니다.
2. 첫 셀(환경 설정)을 실행하고 API 키를 등록합니다.
3. 셀을 위에서부터 차례로 실행하며 책 본문과 견주어 봅니다.
4. 각 장 끝의 심화 실습으로 배운 것을 자신의 문제에 적용합니다.

---

문의·정오표는 이 저장소의 Issues로 남겨 주세요.
