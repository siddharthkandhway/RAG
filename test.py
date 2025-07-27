import google.generativeai as genai

# Use your Gemini API key
genai.configure(api_key="AIzaSyCk3nx8HzrGThZsdfrSHhkpq9zRU2v7Dpo")

# Use the latest stable model
model = genai.GenerativeModel(model_name="models/gemini-2.5-pro")

# Send a text prompt
response = model.generate_content("Summarize the plot of Inception in 3 sentences.")

# Print the output
print(response.text)
