"""Irix v1.0: Basic chatbot using 2 models 'qwen2.5:7b' and ;qwen3:8b' for reasoning with temp history to save chat"""

import ollama

    #Function to check if user's query requires complex reasoning
def ExplicitcheckForHeavyReasoning(usr_inpt: str) -> bool:
    keywords = ("explain", "why", "how", "elaborate", "expand", "prove")
    text = usr_inpt.lower()
    return any(k in text for k in keywords)

    #Handles the chatbot interaction by selecting the appropriate model, generating a response, and updating conversation history.
def chatbot(usr_inpt,lm,hm,history) -> None:
    model = hm if ExplicitcheckForHeavyReasoning(usr_inpt) else lm
        
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
