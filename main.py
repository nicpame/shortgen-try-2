from llm.gemini_client import generate_gemini_response
def main():
    
    response = generate_gemini_response("who are you")
    print(response)


if __name__ == "__main__":
    main()
