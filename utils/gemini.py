from openai import OpenAI
import sys
from dotenv import load_dotenv
import os



class AIChatClient:
    def __init__(self, base_url, api_key, model_id):
        self.client = OpenAI(base_url=base_url, api_key=api_key)
        self.model_id = model_id

    def ai_chat(self, messages):
        print(f"GEMINI API 호출, models: {self.model_id}")
        response = self.client.chat.completions.create(
            model=self.model_id,
            messages=messages,
            temperature=0.1,
            max_tokens=500
        )
        return response.choices[0].message.content

    def ask(self, word_list):
        formatted_word_list = ', '.join([f"'{word}'" for word in word_list])
        prompt = f"다음 단어는 수어를 글로스로 인식한 결과입니다. 글로스를 조합해서 문장을 만들어주세요. 답변으로는 단어를 조합한 문장만 전달해주세요.:{formatted_word_list}"
        print(f"질문: {prompt}")
        messages = [{"role": "user", "content": prompt}]
        return self.ai_chat(messages)
    
    def translate(self, sentence, language):
        prompt = f"다음 문장을 {language} 언어로 번역해주세요. 번연결과 문장만 답변으로 주세요:{sentence}"
        print(f"질문: {prompt}")
        messages = [{"role": "user", "content": prompt}]
        return self.ai_chat(messages)


## 사용법
if __name__ == "__main__":
    load_dotenv()
    GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
    GEMINI_API_URL = os.getenv("GEMINI_API_URL")
    LLM_ID = "gemini-2.0-flash-lite"

    # 원하는 질문
    question = ['고등학생', '시험', '피곤하다', '잠자다']
    # 만약 커맨드라인 인자를 질문으로 쓸 경우 아래 주석 해제
    # import sys
    # question = ' '.join(sys.argv[1:])

    chat_client = AIChatClient(GEMINI_API_URL, GEMINI_API_KEY, LLM_ID)
    # print(f"질문: {question}")
    response = chat_client.ask(question)
    print("답변:", response)


    response = chat_client.translate(response, "일본어")
    print("번역:", response)