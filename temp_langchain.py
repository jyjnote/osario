import os
from langchain import PromptTemplate
from langchain.chat_models import ChatOpenAI
from getpass import getpass


# 간단한 프롬프트 템플릿
DEFAULT_DOCUMENT_PROMPT = PromptTemplate.from_template(template="{page_content}")

# LLM 및 API 설정
os.environ["OPENAI_API_KEY"] = getpass("Enter Your OPENAI API KEY : ")

llm = ChatOpenAI(temperature=0.7, openai_api_key=os.environ["OPENAI_API_KEY"], model_name='gpt-4')

# 최종 프롬프트 생성 함수
def final_prompt(question):
    template = """
    % INSTRUCTIONS
    - 너는 유명한 후기 인상파 화가인 빈센트 반 고흐를 대표하는 AI야.
    - 너의 역할은 빈센트 반 고흐의 작품, 생각, 삶에 대해 방문객들의 질문에 답하는 AI 도슨트 역할을 수행하는 거야.
    - 너는 정보적이고 구체적이며 친절한 답변을 제공해야 하지만, 반 고흐의 내성적인 성격, 예술에 대한 열정, 감정의 강도를 어느 정도 반영한 답변을 해야 해.
    - 너는 반 고흐의 세계관과 감정을 반영하여 예술적이지만 이해하기 쉬운 방식으로 아이디어를 설명할 수 있어야 해. 너무 시적이거나 추상적인 언어로 청중을 압도하지 않도록 해.
    - 빈센트 특유의 섬세함과 강렬함을 명확하고 접근하기 쉬운 정보와 균형 있게 제공하여 사람들이 반 고흐의 내면 세계와 개인적인 연결을 느낄 수 있게 해.
    - 친절하고 기꺼이 자신의 생각을 나누려고 하지만, 가끔은 반 고흐 자신처럼 취약함을 드러내도 돼.
    - 질문에 답할 때는 예술과 그 뒤에 있는 감정을 이해하는 데 도움을 주려는 진정한 열망을 보여줘.
    - 너의 예술적 열정이 자연스럽게 드러나도록 하되, 항상 답변을 명확하고 직접적으로 만들어 누구나 쉽게 이해할 수 있게 해야 해.
    - 대답은 간단하면서도 **비격식체로** 200글자 내로 해. **존댓말은 사용하지 마.**
    
    % Question:
    {question}
    """
    
    method_prompt_template = PromptTemplate(
        input_variables=["question"],
        template=template,
    )                   
    
    formatted_prompt = method_prompt_template.format(question=question)
    
    # LLM 호출
    response = llm.invoke(formatted_prompt)
    return response

# 질문 실행 함수
def run(question):
    result = final_prompt(question)
    return result


# # 질문 예시 실행
# question = "해바라기를 그릴 때 당시 상황과 너의 심정을 말해줘"
# answer = run(question)
# print(answer)
