"""Irix v1.10.0: Changed Irix Structure with complexity classification and web search classification for both Direct and Deliberate paths"""

from IrixAI import IrixSystem

def main() -> None:
    Irix = IrixSystem()
    
    while True:
        user_prompt = input("You: ").strip()
        if user_prompt.lower() in ('exit', 'quit', 'bye'):
            print("Irix: cya👋")
            break
        
        Irix.process(user_prompt)


if __name__ == "__main__":
    main()