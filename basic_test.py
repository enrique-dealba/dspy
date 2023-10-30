import dspy

port = 8888 # maybe not needed
model = "mistralai/Mistral-7B-Instruct-v0.1"
llm = dspy.HFClientTGI(model=model, port=port, max_tokens=250)

dspy.settings.configure(lm=llm)

class RAG(dspy.Module):
    def __init__(self, num_passages=3):
        super().__init__()
        self.retrieve = dspy.Retrieve(k=num_passages)
        self.generate_answer = dspy.ChainOfThought("context, question -> answer")
    
    def forward(self, question):
        context = self.retrieve(question).passages
        answer = self.generate_answer(context=context, question=question)
        return answer
    
rag = RAG()
response = rag("what is the capital of France?").answer  # -> "Paris"
print(f"LLM Response: {response}")
