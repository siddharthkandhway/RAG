import google.generativeai as genai

load_dotenv()
api_key = os.getenv("GOOGLE_API_KEY")
# Use your Gemini API key
genai.configure(api_key=api_key)

model = genai.GenerativeModel('gemini-1.0-pro')
chat = model.start_chat()
response = chat.send_message("Tell me a joke.")
print(response.text)