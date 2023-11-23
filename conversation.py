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
    
    # search top_k & record time
    start_time_similarity_search = time.time()    
    top_k=similarity_search(user_question)
    end_time_similarity_search = time.time()
    
    # parsing top_k
    page_content, metadata = parsing_top_k(top_k)
    
    # feed to gpt & record time
    start_time_feed_ques2gpt = time.time()
    assistant_response = feed_ques2gpt(user_question, page_content)
    end_time_feed_ques2gpt = time.time()

    return  assistant_response


def similarity_search(user_question):
    top_k=docsearch.similarity_search_with_score(query=user_question, k=3)
    return top_k    


def parsing_top_k(top_k):
    """reading information from the top_k results"""
    page_content=[i[0].page_content for i in top_k]
    metadata=[top_k[i][0].metadata for i in range(len(top_k))]
    meta_list = ["/".join(i.values()) for i in metadata]
    return page_content, meta_list

def feed_ques2gpt(user_question, page_content):
    """feed the user_question & top_k result to GPT"""
    # feed to GPT
    response=chat(chat_prompt.format_prompt(
        user_question=user_question,
        relevant_info="\n\n".join(page_content)
    ).to_messages())
    return response.content