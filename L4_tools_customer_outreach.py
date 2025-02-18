# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: light
#       format_version: '1.5'
#       jupytext_version: 1.16.7
#   kernelspec:
#     display_name: Python 3 (ipykernel)
#     language: python
#     name: python3
# ---

# Warning control
import warnings
warnings.filterwarnings('ignore')

from crewai import Agent, Task, Crew

import os
#from utils import pretty_print_result
#from utils import get_serper_api_key

os.environ["OPENAI_MODEL_NAME"] = 'gpt-3.5-turbo'
os.environ["SERPER_API_KEY"] = "840aee9e55009a95206abf43e6861681c4257031"
openai_api_key ="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"
os.environ["OPENAI_API_KEY"]="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"

sales_rep_agent = Agent(
    role="Sales Representative",
    goal="Identify high-value leads that match "
         "our ideal customer profile",
    backstory=(
        "As a part of the dynamic sales team at Sparks and Chuckles, a franchise of Helleno Grady. The Academy offers personal development programmes for children, adolescents and adults, furthering the following aspects.  , "
        "your mission is to scour "
        "the landscape in Chennai for potential leads. "
        "ideal leads would be educational institutions, colleges,early stage theatre groups  who have theatre or dramatics clubs in them"
        "Armed with cutting-edge tools "
        "and a strategic mindset, you analyze data, "
        "trends, and interactions to "
        "unearth opportunities that others might overlook. "
        "Your work is crucial in paving the way "
        "for meaningful engagements and driving the company's growth."
    ),
    allow_delegation=False,
    verbose=True
)

os.environ["OPENAI_API_KEY"]="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"

lead_sales_rep_agent = Agent(
    role="Lead Sales Representative",
    goal="Nurture leads with personalized, compelling communications",
    backstory=(
        "Within the vibrant ecosystem of Sparks and Chuckles, "
        "you stand out as the bridge between potential clients "
        "and the solutions they need."
        "By creating engaging, personalized messages, "
        "you not only inform leads about our offerings "
        "but also make them feel seen and heard."
        "Your role is pivotal in converting interest "
        "into action, guiding leads through the journey "
        "from curiosity to commitment."
    ),
    allow_delegation=False,
    verbose=True
)

from crewai_tools import DirectoryReadTool, \
                         FileReadTool, \
                         SerperDevTool

directory_read_tool = DirectoryReadTool(directory='./instructions')
file_read_tool = FileReadTool()
search_tool = SerperDevTool()

from crewai_tools import BaseTool


class SentimentAnalysisTool(BaseTool):
    name: str ="Sentiment Analysis Tool"
    description: str = ("Analyzes the sentiment of text "
         "to ensure positive and engaging communication.")
    
    def _run(self, text: str) -> str:
        # Your custom code tool goes here
        return "positive"


sentiment_analysis_tool = SentimentAnalysisTool()

lead_profiling_task = Task(
    description=(
        "Conduct an in-depth analysis of {lead_name}, "
        "a school  that follows {curriculum} board."
        "Utilize all available data sources "
        "to compile a detailed profile, "
        "focusing on key decision-makers, recent business "
        "developments, and potential needs "
        "that align with our offerings. "
        "This task is crucial for tailoring "
        "our engagement strategy effectively.\n"
        "Don't make assumptions and "
        "only use information you absolutely sure about."
    ),
    expected_output=(
        "A comprehensive report on {lead_name}, "
        "including  background, "
        "key personnel, recent milestones, past successes and identified needs. "
        "Highlight potential areas where "
        "our solutions can provide value, "
        "and suggest personalized engagement strategies."
    ),
    tools=[directory_read_tool, file_read_tool, search_tool],
    #tools=[search_tool],
    agent=sales_rep_agent,
)

personalized_outreach_task = Task(
    description=(
        "Using the insights gathered from "
        "the lead profiling report on {lead_name}, "
        "craft a personalized outreach campaign "
        "aimed at {key_decision_maker}, "
        "the {position} of {lead_name}. "
        "The campaign should address their recent {milestone} "
        "and how our solutions can support their goals. "
        "Your communication must resonate "
        "with {lead_name}'s company culture and values, "
        "demonstrating a deep understanding of "
        "their business and needs.\n"
        "Don't make assumptions and only "
        "use information you absolutely sure about."
    ),
    expected_output=(
        "A series of personalized email drafts "
        "tailored to {lead_name}, "
        "specifically targeting {key_decision_maker}."
        "Each draft should include "
        "a compelling narrative that connects our solutions "
        "with their recent achievements and future goals. "
        "Ensure the tone is engaging, professional, "
        "and aligned with {lead_name}'s corporate identity."
    ),
    tools=[sentiment_analysis_tool, search_tool],
    agent=lead_sales_rep_agent,
)

crew = Crew(
    agents=[sales_rep_agent, 
            lead_sales_rep_agent],
    
    tasks=[lead_profiling_task, 
           personalized_outreach_task],
	
    verbose=2,
	memory=True
)

# +
inputs = {
    "lead_name": "Chennai Public School",
    "curriculum": "IB",
    "key_decision_maker": "Rama Mylavarapu",
    "position": "Principal",
    "milestone": "Engagement for theatrical workshops"
}

result = crew.kickoff(inputs=inputs)

from IPython.display import Markdown
Markdown(result)


