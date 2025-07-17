from llm.gemini_client import list_available_gemini_models, generate_text, generate_image_and_save_file
def main():
    
    # print(generate_text(
    #     prompt="Hello, how are you?",
    #     model_key="flash"))
    # print(generate_image_and_save_file('cat reading book', './image.jpg'))
    list_available_gemini_models()

if __name__ == "__main__":
    main()
