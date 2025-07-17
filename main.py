from llm.gemini_client import list_available_gemini_models, generate_text
def main():
    
    print(generate_text(
        prompt="Hello, how are you?",
        model_key="flash"))

if __name__ == "__main__":
    main()
