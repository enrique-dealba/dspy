import dspy
from dspy.evaluate import Evaluate
from dspy.datasets.gsm8k import GSM8K, gsm8k_metric
from dspy.teleprompt import BootstrapFewShotWithRandomSearch

port = 8888 # maybe not needed
model = "mistralai/Mistral-7B-Instruct-v0.1"
llm = dspy.HFClientTGI(model=model, port=port, max_tokens=250)

# CoT Testing
gms8k = GSM8K()
trainset, devset = gms8k.train, gms8k.dev

dspy.settings.configure(lm=llm)

NUM_THREADS = 2 # prev: 4
evaluate = Evaluate(devset=devset[:], metric=gsm8k_metric, num_threads=NUM_THREADS, display_progress=True, display_table=0)


class CoT(dspy.Module):
    def __init__(self):
        super().__init__()
        self.prog = dspy.ChainOfThought("question -> answer")
    
    def forward(self, question):
        return self.prog(question=question)

RUN_FROM_SCRATCH = False

if RUN_FROM_SCRATCH:
    config = dict(max_bootstrapped_demos=8, max_labeled_demos=8, num_candidate_programs=10, num_threads=NUM_THREADS)
    teleprompter = BootstrapFewShotWithRandomSearch(metric=gsm8k_metric, **config)
    cot_bs = teleprompter.compile(CoT(), trainset=trainset, valset=devset)
    # cot_bs.save('mistral_8_8_10_2_gsm8k_200_300.json')
else:
    cot_bs = CoT()
    cot_bs.load('mistral_8_8_10_2_gsm8k_200_300.json')

evaluate(cot_bs, devset=devset[:])

llm.inspect_history(n=1)

print("Finished CoT for Mistral LLM")
