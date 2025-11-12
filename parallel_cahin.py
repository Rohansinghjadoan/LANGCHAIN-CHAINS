from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint
from dotenv import load_dotenv
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser 
from langchain_core.runnables import RunnableParallel




load_dotenv()

llm1= HuggingFaceEndpoint(
    repo_id="google/gemma-2-2b-it",
    task="text-generation"
)
model1=ChatHuggingFace(llm=llm1)

llm2=HuggingFaceEndpoint(
    repo_id='google/gemma-2-2b-it',## can use diffrent model
    task='text-generation'
)

model2=ChatHuggingFace(llm=llm2)
prompt1 = PromptTemplate(
    template="Generate short and simple notes from the following text:\n{text}",
    input_variables=["text"]
)

prompt2 = PromptTemplate(
    template="Generate 5 short question answers from the following text:\n{text}",
    input_variables=["text"]
)

prompt3 = PromptTemplate(
    template="Merge the provided notes and quiz into a single document.\nNotes: {notes}\nQuiz: {quiz}",
    input_variables=["notes", "quiz"]
)

# 3. Output parser
parser = StrOutputParser()
parallel_chain = RunnableParallel({
    "notes": prompt1 | model1 | parser,
    "quiz": prompt2 | model2 | parser
})

# 5. Merge chain
merge_chain = prompt3 | model1 | parser

# 6. Combine
chain = parallel_chain | merge_chain

# 7. Input text
text = """
Support vector machines (SVMs) are supervised learning methods used for classification, regression, and outlier detection.
They work well in high-dimensional spaces, use support vectors for efficiency, and allow various kernel functions.
However, they may overfit when features > samples, and computing probability estimates can be costly.
"""

# 8. Run chain
result = chain.invoke({"text": text})
print(result)

# 9. Optional: visualize pipeline
chain.get_graph().print_ascii()
