from langchain_google_genai import ChatGoogleGenerativeAI

def get_llm(llm_model, temperature, **llm_kwargs):
    """
    Initializes and returns an LLM instance, handling different configurations.
    """
    if hasattr(llm_model, 'llm_model'):
        config = llm_model
        model_name = config.llm_model or "gemini-2.0-flash-lite"
        temp = config.temperature or 0.0
        if hasattr(config, 'google_api_key') and config.google_api_key:
            llm_kwargs['google_api_key'] = config.google_api_key
    else:
        model_name = llm_model
        temp = temperature

    return ChatGoogleGenerativeAI(
        model=model_name,
        temperature=temp,
        **llm_kwargs
    )
