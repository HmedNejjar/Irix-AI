# type: ignore
import sys
from lm_eval.api.model import LM
from lm_eval.api.registry import register_model
from lm_eval import simple_evaluate
from lm_eval.tasks import get_task_dict
import lm_eval.utils as utils
from IrixAI import IrixSystem
import pandas as pd

@register_model("irix_pipeline")
class IrixEvalWrapper(LM):
    def __init__(self):
        super().__init__()
        # Initialize your system
        self.irix = IrixSystem()
        # Save the base system prompt to reset history later
        self.base_sys_prompt = self.irix.history[0] 
        
    def generate_until(self, requests):
        """
        This method is called by the harness for generative tasks.
        """
        res = []
        for request in requests:
            # request.args[0] contains the actual prompt text
            prompt = request.args[0]
            
            # 1. PURGE HISTORY: Ensure each benchmark question is isolated
            # We only keep the very first element (the system prompt)
            self.irix.history = [self.base_sys_prompt]
            
            # 2. RUN IRIX-AI: Process the prompt through your router & agents
            try:
                # process() returns the final bot_answer_content
                output = self.irix.process(prompt) 
            except Exception as e:
                print(f"Error processing prompt: {e}")
                output = ""
                
            res.append(output)
            
        return res
        
    def loglikelihood(self, requests):
        """Irix-AI generates text; it does not output raw log probabilities."""
        raise NotImplementedError("Loglikelihood not supported by Irix-AI MAS.")
        
    def loglikelihood_rolling(self, requests):
        raise NotImplementedError("Rolling loglikelihood not supported.")

# Run the evaluation
if __name__ == "__main__":
    print("Starting evaluation of Irix-AI Pipeline...")
    
    tasks = ["gsm8k"]
    irix_wrapper = IrixEvalWrapper()
    
    # Remove task_dict=task_dict from the call below
    results = simple_evaluate(
        model=irix_wrapper,
        tasks=tasks,
        num_fewshot=0,
        limit=10
    )

    # --- EXPORT TO EXCEL LOGIC ---
    export_data = []
    
    for task_name, metrics in results["results"].items():
        row = {"Task": task_name}
        row.update(metrics) 
        export_data.append(row)
    
    df = pd.DataFrame(export_data)
    filename = "irix_eval_results.xlsx"
    df.to_excel(filename, index=False)
    
    print(f"\n✅ Evaluation complete! Results saved to {filename}")
    
    print(utils.make_table(results))