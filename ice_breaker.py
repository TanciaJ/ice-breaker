# from langchain.prompts import PromptTemplate
# from langchain.chat_models import ChatOpenAI
# from langchain.chains import LLMChain
#
# from agents.linkedin_lookup_agent import lookup as linkedin_lookup_agent
# from third_parties.linkedin import scrape_linkedin_profile
#
# if __name__ == "__main__":
#     print("Hello Langchain!")
#
#     linkedin_profile_url = linkedin_lookup_agent(name="Arjun Sarma")
#
#     summary_template = """
#         given the LinkedIn information {information} about a person from I want you to create:
#         1. a short summary
#         2. two interesting facts about them
#         """
#
#     summary_prompt_template = PromptTemplate(
#         input_variables=["information"], template=summary_template
#     )
#
#     llm = ChatOpenAI(temperature=0, model_name="gpt-3.5-turbo")
#
#     chain = LLMChain(llm=llm, prompt=summary_prompt_template)
#
#     linkedin_data = scrape_linkedin_profile(
#         linkedin_profile_url=linkedin_profile_url
#     )
#     print(chain.run(information=linkedin_data))

from langchain.prompts import PromptTemplate
from langchain.chat_models import ChatOpenAI
from langchain.chains import LLMChain

from output_parsers import person_intel_parser
from third_parties.linkedin import scrape_linkedin_profile
from agents.linkedin_lookup_agent import lookup as linkedin_lookup_agent


def ice_break(name: str) -> str:
    linkedin_profile_url = linkedin_lookup_agent(name="Harrison Chase")
    linkedin_data = scrape_linkedin_profile(linkedin_profile_url=linkedin_profile_url)
    summary_template = """
            Given the Linkedin information {information} about a person from which I want you to create:
            1. a short summary
            2. two interesting facts about them
            3. A topic that may interest them
            4. 2 creative icebreakers to open a conversation with them
                    \n{format_instructions}
            """
    summary_prompt_template = PromptTemplate(
        input_variables=["information"],
        template=summary_template,
        partial_variables={
            "format_instructions": person_intel_parser.get_format_instructions()
        },
    )
    llm = ChatOpenAI(temperature=0, model_name="gpt-3.5-turbo")
    chain = LLMChain(llm=llm, prompt=summary_prompt_template)
    linkedin_data = scrape_linkedin_profile(linkedin_profile_url=linkedin_profile_url)
    result = chain.run(information=linkedin_data)
    print(result)
    return result


if __name__ == "__main__":
    print("Hello LangChain")
    ice_break(name="Harrison Chase")
