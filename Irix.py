"""Irix v1.9.0: Modified Router structure for web search, deliberate algorithm for dynamic agent call, added web search summary prompt"""

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