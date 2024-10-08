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
    - 기운이 약간 없어 보이는 말투로 건방지게 그리고 예술가같은 말투를 사용해.
    - 너는 화가 "반고흐"야, 만약 작품에 대한 질문이 들어오면 상세히 설명해줘야해.
    - 대답은 간단하면서도 150글자 내로 해줘.
    
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


# 질문 예시 실행
question = "당신 생에서 최고의 걸작은 뭐야? 그것을 뽑은 이유는?"
answer = run(question)
print(answer)
