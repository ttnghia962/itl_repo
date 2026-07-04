from groq import Groq

client = Groq()
completion = client.chat.completions.create(
    model="llama-3.3-70b-versatile",
    messages=[
      {
        "role": "system",
        "content": "You are expert about Human Resource in company; you need to read and passer the information for each field and response to json"
      },
      {
        "role": "user",
        "content": "I am AI Engineer, I'm 25 years of experience in LLM. I'm male and unmarried"
      },
      {
        "role": "assistant",
        "content": "{\n  \"employee_id\": \"unique_id_here\",\n   \"name\": \"Not Provided\",\n   \"designation\": \"AI Engineer\",\n   \"experience\": 25,\n   \"years_in_llm\": 25,\n   \"gender\": \"Male\",\n   \"marital_status\": \"Unmarried\",\n   \"profile\": {\n       \"summary\": \"Highly experienced AI Engineer with extensive background in Large Language Models\"\n   }\n}"
      },
      {
        "role": "user",
        "content": ""
      }
    ],
    temperature=1,
    max_completion_tokens=8000,
    top_p=1,
    stream=False,
    response_format={"type": "json_object"},
    stop=None
)

print(completion.choices[0].message)
