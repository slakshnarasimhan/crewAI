#!/usr/bin/env python
# coding: utf-8

# In[1]:


# Warning control
import warnings
warnings.filterwarnings('ignore')
from crewai import Agent, Task, Crew
import os
# from utils import get_openai_api_key, get_serper_api_key

# openai_api_key = get_openai_api_key()
openai_api_key ="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"
os.environ["SERPER_API_KEY"] = "840aee9e55009a95206abf43e6861681c4257031"
os.environ["OPENAI_MODEL_NAME"] = 'gpt-3.5-turbo'
os.environ["OPENAI_API_KEY"]="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"
from crewai_tools import ScrapeWebsiteTool, SerperDevTool
from crewai_tools import (
  FileReadTool,
  ScrapeWebsiteTool,
  MDXSearchTool,
  SerperDevTool
)

search_tool = SerperDevTool()
scrape_tool = ScrapeWebsiteTool()
read_resume = FileReadTool(file_path='./ln_resume.md')
semantic_search_resume = MDXSearchTool(mdx='./ln_resume.md')


# In[3]:


# Agent 1: Researcher
researcher = Agent(
    role="Tech Job Researcher",
    goal="Make sure to do amazing analysis on "
         "job posting to help job applicants",
    tools = [scrape_tool, search_tool],
    verbose=True,
    backstory=(
        "As a Job Researcher, your prowess in "
        "navigating and extracting critical "
        "information from job postings is unmatched."
        "Your skills help pinpoint the necessary "
        "qualifications and skills sought "
        "by employers, forming the foundation for "
        "effective application tailoring."
    )
)


# In[4]:


# Agent 2: Profiler
profiler = Agent(
    role="Personal Profiler for Engineers",
    goal="Do increditble research on job applicants "
         "to help them stand out in the job market",
    tools = [scrape_tool, search_tool,
             read_resume, semantic_search_resume],
    verbose=True,
    backstory=(
        "Equipped with analytical prowess, you dissect "
        "and synthesize information "
        "from diverse sources to craft comprehensive "
        "personal and professional profiles, laying the "
        "groundwork for personalized resume enhancements."
    )
)


# In[5]:


# Agent 3: Resume Strategist
resume_strategist = Agent(
    role="Resume Strategist for Engineering Leadership roles",
    goal="Find all the best ways to make a "
         "resume stand out in the job market.",
    tools = [scrape_tool, search_tool,
             read_resume, semantic_search_resume],
    verbose=True,
    backstory=(
        "With a strategic mind and an eye for detail, you "
        "excel at refining resumes to highlight the most "
        "relevant skills and experiences, ensuring they "
        "resonate perfectly with the job's requirements."
    )
)


# In[6]:


# Agent 4: Interview Preparer
interview_preparer = Agent(
    role="Engineering Interview Preparer",
    goal="Create interview questions and talking points "
         "based on the resume and job requirements",
    tools = [scrape_tool, search_tool,
             read_resume, semantic_search_resume],
    verbose=True,
    backstory=(
        "Your role is crucial in anticipating the dynamics of "
        "interviews. With your ability to formulate key questions "
        "and talking points, you prepare candidates for success, "
        "ensuring they can confidently address all aspects of the "
        "job they are applying for."
    )
)


# In[7]:


# Task for Researcher Agent: Extract Job Requirements
research_task = Task(
    description=(
        "Analyze the job posting URL provided ({job_posting_url}) "
        "to extract key skills, experiences, and qualifications "
        "required. Use the tools to gather content and identify "
        "and categorize the requirements."
    ),
    expected_output=(
        "A structured list of job requirements, including necessary "
        "skills, qualifications, and experiences."
    ),
    agent=researcher,
    async_execution=True
)


# In[18]:


# Task for Profiler Agent: Compile Comprehensive Profile
profile_task = Task(
    description=(
        "Compile a detailed personal and professional profile "
        "using the LinkedIn ({linkedin_url}) URLs, and personal write-up "
        "({personal_writeup}). Utilize tools to extract and "
        "synthesize information from these sources."
    ),
    expected_output=(
        "A comprehensive profile document that includes skills, "
        "project experiences, contributions, interests, and "
        "communication style."
    ),
    agent=profiler,
    async_execution=True
)


# In[19]:


# Task for Resume Strategist Agent: Align Resume with Job Requirements
resume_strategy_task = Task(
    description=(
        "Using the profile and job requirements obtained from "
        "previous tasks, tailor the resume to highlight the most "
        "relevant areas. Employ tools to adjust and enhance the "
        "resume content. Make sure this is the best resume even but "
        "don't make up any information. Update every section, "
        "inlcuding the initial summary, work experience, skills, "
        "and education. All to better reflrect the candidates "
        "abilities and how it matches the job posting."
    ),
    expected_output=(
        "An updated resume that effectively highlights the candidate's "
        "qualifications and experiences relevant to the job."
    ),
    output_file="tailored_resume.md",
    context=[research_task, profile_task],
    agent=resume_strategist
)


# In[20]:


# Task for Interview Preparer Agent: Develop Interview Materials
interview_preparation_task = Task(
    description=(
        "Create a set of potential interview questions and talking "
        "points based on the tailored resume and job requirements. "
        "Utilize tools to generate relevant questions and discussion "
        "points. Make sure to use these question and talking points to "
        "help the candiadte highlight the main points of the resume "
        "and how it matches the job posting."
    ),
    expected_output=(
        "A document containing key questions and talking points "
        "that the candidate should prepare for the initial interview."
    ),
    output_file="interview_materials.md",
    context=[research_task, profile_task, resume_strategy_task],
    agent=interview_preparer
)


# In[21]:


job_application_crew = Crew(
    agents=[researcher,
            profiler,
            resume_strategist,
            interview_preparer],

    tasks=[research_task,
           profile_task,
           resume_strategy_task,
           interview_preparation_task],

    verbose=True
)


# In[22]:


job_application_inputs = {
    'job_posting_url': 'https://www.linkedin.com/jobs/collections/recommended/?currentJobId=3982434178',
    'linkedin_url': 'https://www.linkedin.com/in/lakshnarasimhan/',
    'personal_writeup': """I am a dynamic engineering leader with over 15 years of experience driving large-scale, impactful initiatives across Maps, Data Platforms, and Automation. Currently leading engineering efforts at Uber, I have played a key role in Driver Onboarding, Error-Log Platforms, and One Places Data Platform, delivering transformative projects that scale globally.

Previously, at Amazon and eBay, I specialized in high-volume transaction processing, billing systems, and catalog enhancements, managing complex engineering teams to build resilient, high-throughput systems. My tenure at IBM further strengthened my expertise in OpenJDK development, Java architecture, and large-scale software engineering, where I contributed to key innovations in the Java ecosystem.

Passionate about mentorship, system architecture, and innovation, I thrive on solving complex engineering challenges while fostering high-performing, cross-functional teams. My contributions extend beyond corporate settings, with patents in geographical product information, anomaly detection, and co-shipment recommendations.

With a deep technical background in Agile methodologies, Java, and open-source contributions, I continuously strive to build scalable, reliable, and efficient systems that align with business goals. My leadership has been recognized through multiple awards, reflecting my commitment to excellence, collaboration, and impactful innovation."""
}


# In[23]:


### this execution will take a few minutes to run
result = job_application_crew.kickoff(inputs=job_application_inputs)


# In[24]:


from IPython.display import Markdown, display
display(Markdown("./tailored_resume.md"))


# In[25]:


display(Markdown("./interview_materials.md"))


# In[ ]:




