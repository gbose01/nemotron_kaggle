import re
from collections import Counter
from concurrent.futures import ThreadPoolExecutor, as_completed

def extract_boxed_answer(text):
    """
    Helper to extract the LaTeX boxed answer.
    """
    if not text:
        return ""
    # Look for \boxed{...}
    match = re.search(r'\\boxed{([^{}]*(?:{[^{}]*}[^{}]*)*)}', text)
    if match:
        return match.group(1).strip()
    match = re.search(r'\\boxed{(.*?)}', text, re.DOTALL)
    if match:
        return match.group(1).strip()
    return ""

class ConsensusInference:
    def __init__(self, model_caller_fn, num_samples=5):
        """
        model_caller_fn: a function that takes a prompt and returns a raw string response from the LLM.
        num_samples: group size (N) for self-consistency voting.
        """
        self.model_caller = model_caller_fn
        self.num_samples = num_samples

    def get_prediction(self, prompt):
        """
        Runs parallel queries to generate multiple reasoning paths, extracts their answers, and performs majority voting.
        """
        # If we only require 1 sample, skip voting to save tokens/calls
        if self.num_samples <= 1:
            raw_response = self.model_caller(prompt)
            return extract_boxed_answer(raw_response), [raw_response]

        responses = []
        # Execute API calls in parallel to maximize speed
        with ThreadPoolExecutor(max_workers=self.num_samples) as executor:
            futures = [executor.submit(self.model_caller, prompt) for _ in range(self.num_samples)]
            for future in as_completed(futures):
                try:
                    res = future.result()
                    responses.append(res)
                except Exception as e:
                    print(f"Sample generation failed: {e}")
                    
        # Extract answers from all generated chains
        answers = []
        for r in responses:
            ans = extract_boxed_answer(r)
            if ans:
                answers.append(ans)
                
        if not answers:
            return "", responses
            
        # Tally votes
        votes = Counter(answers)
        majority_ans, count = votes.most_common(1)[0]
        
        print(f"Consensus Votes: {dict(votes)} | Winner: {majority_ans} ({count}/{len(responses)} votes)")
        return majority_ans, responses
