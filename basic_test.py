import dspy

port = 8888 # maybe not needed
model = "mistralai/Mistral-7B-Instruct-v0.1"
llm = dspy.HFClientTGI(model=model, port=port, max_tokens=150)

dspy.settings.configure(lm=llm)

class BasicQA(dspy.Signature):
    """Answer questions with short factoid answers."""

    question = dspy.InputField()
    answer = dspy.OutputField(desc="often between 1 and 5 words")
    
# Define the predictor.
generate_answer = dspy.Predict(BasicQA)

# Call the predictor on a particular input.
example_question = "At My Window was released by which American singer-songwriter?"
example_answer = "John Townes Van Zandt"

pred = generate_answer(question=example_question)

# Print the input and the prediction.
print(f"Question: {example_question}")
print(f"Predicted Answer: {pred.answer}")
print(f"Actual Answer: {example_answer}")
