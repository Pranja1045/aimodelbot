import google import genai

# Configure your API key
genai.configure(api_key="AIzaSyDS8ErPHYNOH1pF99BolVe4vdv0JEgmaCg")

# Create a model instance
model = genai.GenerativeModel('gemini-2.5-flash')

# Generate content
response = model.generate_content("Hello, how are you?")
print(response.text)
