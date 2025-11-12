from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint
from dotenv import load_dotenv
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser, PydanticOutputParser
from langchain_core.runnables import RunnableBranch, RunnableLambda
from pydantic import BaseModel, Field
from typing import Literal

load_dotenv()

# ---- Model ----
llm = HuggingFaceEndpoint(
    repo_id="google/gemma-2-2b-it",
    task="text-generation"
)
model = ChatHuggingFace(llm=llm)
str_parser = StrOutputParser()

# ---- Schema ----
class Feedback(BaseModel):
    sentiment: Literal["positive", "negative"] = Field(
        description="Sentiment of the feedback (positive or negative)"
    )

parser = PydanticOutputParser(pydantic_object=Feedback)

# ---- Classifier prompt (escaped braces!) ----
prompt_classifier = PromptTemplate(
    template=(
        "You are a sentiment classifier.\n"
        "Classify the following feedback as either 'positive' or 'negative'.\n\n"
        "Feedback: {feedback}\n\n"
        "Return the output STRICTLY as a JSON object in one of these two forms:\n"
        '{{"sentiment": "positive"}} OR {{"sentiment": "negative"}}\n'
        "Do not include any explanation or text outside of the JSON."
    ),
    input_variables=["feedback"],
)

classifier_chain = prompt_classifier | model | parser

# ---- Prompts for responses ----
prompt_positive = PromptTemplate(
    template="Write an appreciative response to this positive feedback:\n{feedback}",
    input_variables=["feedback"]
)

prompt_negative = PromptTemplate(
    template="Write a polite, empathetic response to this negative feedback:\n{feedback}",
    input_variables=["feedback"]
)

# ---- Branch logic ----
branch_chain = RunnableBranch(
    (lambda x: x.sentiment == "positive", prompt_positive | model | str_parser),
    (lambda x: x.sentiment == "negative", prompt_negative | model | str_parser),
    RunnableLambda(lambda _: "Could not determine sentiment.")
)

# ---- Full chain ----
chain = classifier_chain | branch_chain

# ---- Test 1 ----
print("🟢 Positive feedback:")
res1 = chain.invoke({"feedback": "This phone looks and feels amazing!"})
print(res1)

# ---- Test 2 ----
print("\n🔴 Negative feedback:")
res2 = chain.invoke({"feedback": "This phone is terrible and crashes constantly."})
print(res2)
