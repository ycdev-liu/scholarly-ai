from openai import OpenAI
 
client = OpenAI(
     base_url="http://127.0.0.1:8045/v1",
     api_key="sk-afdca1cf4e964a6a8c895f0eecfa905e"
 )
 
response = client.chat.completions.create(
     model="gemini-3-flash",
     messages=[{"role": "user", "content": "Hello"}]
 )
 
print(response.choices[0].message.content)