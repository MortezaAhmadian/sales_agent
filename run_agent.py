from openai import OpenAI

client = OpenAI()

def run_agent(prospect_name: str, company_name: str) -> str:
    messages = [
        {
            "role": "system", 
            "content": """You are a sales research agent. Given a prospect's name and company,
                        use your tools to research them thoroughly. Then return a JSON briefing with:
                        - company_summary (2-3 sentences)
                        - recent_news (list of 3 bullet points)
                        - likely_pain_points (list)
                        - suggested_talking_points (list)
                        - risks_or_objections (list)"""
            },
        {
            "role": "user", 
            "content": f"Research {prospect_name} at {company_name}"
            }
    ]

    while True:
        response = client.chat.completions.create(
            model="gpt-4o",
            messages=messages,
            tools=tools,
            tool_choice="auto"
        )
        msg = response.choices[0].message
        messages.append(msg)
       