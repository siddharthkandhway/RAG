import google.generativeai as genai

# Set your API key
genai.configure(api_key="AIzaSyCk3nx8HzrGThZsdfrSHhkpq9zRU2v7Dpo")

# List all available models
models = genai.list_models()

# Print model names and descriptions
for model in models:
    print(f"Model Name: {model.name}")
    print(f"  Description: {model.description}")
    print(f"  Input Types: {model.supported_generation_methods}")
    print()
