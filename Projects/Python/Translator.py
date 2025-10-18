from googletrans import Translator
from langdetect import detect

translator = Translator()

async def auto_translate(text, target_lang='en'):
    source_lang = detect(text)
    if source_lang != target_lang:
        translation = await translator.translate(text, dest=target_lang)
        return translation.text
    return text

# Build multilingual chatbot
async def multilingual_response(user_message):
    # Detect language
    user_lang = detect(user_message)
    
    # Translate to English for processing
    english_msg = await auto_translate(user_message, 'en')
    
    # Generate response
        # define function to generate response and then  
    # Translate back
    # final_response = auto_translate(response, user_lang)
    return english_msg

async def main():


    en = await multilingual_response(input())

    print(en)

if __name__ == '__main__':
    import asyncio
    asyncio.run(main())
