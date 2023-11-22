from langchain import OpenAI, ConversationChain, LLMChain, PromptTemplate
from langchain.prompts.chat import (
        ChatPromptTemplate,
        SystemMessagePromptTemplate,
        AIMessagePromptTemplate,
        HumanMessagePromptTemplate,
)

load_dotenv()

chat = ChatOpenAI(temperature=0)

def conversation(user_question):
    system_template="""You are an expert about policies of ITL Corporation, I will ask you a question, and then provide you some chunks of text contain relevant information. 
    Try to extract information from the provided text & answer in Vietnamese. 
    You should answer straight to the point of the question, ignore irrelevant information, prefer bullet-points. 
    If the text does not contain relevant information, you should tell me that you don't have the answer.
    """
    human_template="""
    Questions:
    {user_question}  

    Relevant Information:
    {relevant_info}
    """
    system_message_prompt = SystemMessagePromptTemplate.from_template(system_template)
    human_message_prompt = HumanMessagePromptTemplate.from_template(human_template)
    chat_prompt = ChatPromptTemplate.from_messages([system_message_prompt, human_message_prompt])
    


    _response = chat(chat_prompt.format_prompt(
        user_question = user_question,
        relevant_info = "\n\n".join(page_content)
    ).to_messages())

    return _response.content
