"""Irix v1.8.2: Changed heavy agent, summary trigger and routing structure"""

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