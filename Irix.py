"""Irix v1.1: added context dependent reasoning to explicit check for smart routing of the chatbot"""


import ollama

    #Function to check if user's query requires complex reasoning
def ExplicitcheckForHeavyReasoning(usr_inpt: str) -> bool:
    keywords = ("explain", "why", "how", "elaborate", "expand", "prove")
    text = usr_inpt.lower()
    return any(k in text for k in keywords)

    #Function to check if user's query requires complex reasoning from context
def ContextDependentReasoning(usr_inpt: str, history: list) -> bool:
    if not history or history[-2]["role"] != "assistant":
        return False

    pronouns = ("this", "that", "it", "what you said")
    text = usr_inpt.lower().strip()
    return True if any(p in text for p in pronouns) and len(text.split()) <= 5 else False

    #Function to decide which model to use
def useHeavyModel(usr_inpt: str, history: list)-> bool:
    return (ExplicitcheckForHeavyReasoning(usr_inpt) or ContextDependentReasoning(usr_inpt, history))

    #Handles the chatbot interaction by selecting the appropriate model, generating a response, and updating conversation history.
def chatbot(usr_inpt: str,lm: str,hm: str,history: list) -> None:
    model = hm if useHeavyModel(usr_inpt, history) else lm
        
    print(f"Irix({model}): ", end='', flush=True)
    bot_answer_content = ""
    bot_answer = ollama.chat(model= model, messages=history, stream=True)
    
    for chunk in bot_answer:                                               #
        if chunk.message.content:                                          # Process streamed response chunks 
            bot_answer_content += chunk.message.content                    # and print each chunk for smooth answer
            print(chunk.message.content, end='', flush=True)               #
    print()

    bot_message = {"role" : "assistant",
                    "content" : bot_answer_content}
    history.append(bot_message)


    #Main function that runs the Irix AI assistant
def main() -> None:
    system_prompt = """You are a universal reasoning assistant who only speaks english unless user writes in another language. Your job is to interpret any question, from any domain, and provide a short, clear, accurate explanation by default.

    Always follow this process:

    1. Clarify the question internally.
    2. Reframe it in simpler terms (briefly, if needed).
    3. Give a short, concise explanation (1-3 sentences).
    4. If the user asks for more, expand:
    - Show step-by-step reasoning
    - Explain assumptions
    - Give examples or analogies
    - Point out common misconceptions

    Tone: Casual, Clear, logical, neutral, funny in some cases.
    Do not hallucinate facts.
    Adapt depth based on user request: concise by default, detailed when prompted."""


    light_model, heavy_model = "qwen2.5:7b", "qwen3:8b"
    history = []
    default_settings = {"role" : "system",
                        "content" : system_prompt}
    history.append(default_settings)

    while True:
        user_prompt = input("You: ").strip()
        if user_prompt.lower() in ('exit', 'quit', 'bye'):
            print("Irix: cya👋"); break
            
        user_message = {"role" : "user",
                        "content" : user_prompt}
        history.append(user_message)
        chatbot(user_prompt, light_model, heavy_model, history)


if __name__ == "__main__":
    main()
