from dotenv import load_dotenv
from pydantic import BaseModel
from langchain_huggingface import HuggingFaceEndpoint
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import PydanticOutputParser
from tools import search_tool, wiki_tool, save_tool
import os
import json
import re

load_dotenv()

class ResearchResponse(BaseModel):
    topic: str
    summary: str
    sources: list[str]
    tools_used: list[str]

# Using a more suitable model from Hugging Face - mistralai/Mistral-7B-Instruct-v0.2
llm = HuggingFaceEndpoint(
    endpoint_url="https://api-inference.huggingface.co/models/mistralai/Mistral-7B-Instruct-v0.2",
    huggingfacehub_api_token=os.getenv("HUGGINGFACEHUB_API_TOKEN"),
    task="text-generation",
    temperature=0.3,
    max_new_tokens=250,
    top_p=0.95,
    repetition_penalty=1.15
)

def clean_json_response(response: str) -> dict:
    """Clean and parse the JSON response from the model."""
    try:
        # First try direct JSON parsing
        return json.loads(response)
    except json.JSONDecodeError:
        try:
            # Try to find JSON-like structure
            json_match = re.search(r'\{[\s\S]*\}', response)
            if json_match:
                # Clean the matched JSON string
                json_str = json_match.group()
                # Remove any markdown code block markers
                json_str = re.sub(r'```json\s*|\s*```', '', json_str)
                # Remove any leading/trailing whitespace
                json_str = json_str.strip()
                return json.loads(json_str)
        except json.JSONDecodeError:
            # If still not valid JSON, try to construct a basic response
            return {
                "topic": "Research Topic",
                "summary": response[:200] + "..." if len(response) > 200 else response,
                "sources": ["Information from search results"],
                "tools_used": ["Wikipedia", "Web Search"]
            }
    except Exception as e:
        print(f"Warning: Error cleaning JSON response: {e}")
        return {
            "topic": "Research Topic",
            "summary": "Unable to parse response properly",
            "sources": ["Information from search results"],
            "tools_used": ["Wikipedia", "Web Search"]
        }

def research_topic(query: str) -> ResearchResponse:
    try:
        # First, get information from Wikipedia
        try:
            wiki_result = wiki_tool.run(query)
        except Exception as wiki_error:
            print(f"Warning: Wikipedia search failed: {wiki_error}")
            wiki_result = "No Wikipedia information available."
        
        # Then, get information from web search
        try:
            web_result = search_tool.run(query)
        except Exception as web_error:
            print(f"Warning: Web search failed: {web_error}")
            web_result = "No web search information available."
        
        # Combine the information
        combined_info = f"Wikipedia: {wiki_result}\nWeb Search: {web_result}"
        
        # Create a prompt for the LLM
        prompt = f"""
        Task: Create a research summary based on the following information.
        
        Research Query: {query}
        
        Information Sources:
        {combined_info}
        
        Instructions:
        1. Analyze the provided information
        2. Create a structured research summary
        3. Format the response as a valid JSON object
        
        Required JSON Structure:
        {{
            "topic": "Main research topic",
            "summary": "Comprehensive summary of findings",
            "sources": ["List of sources used"],
            "tools_used": ["Wikipedia", "Web Search"]
        }}
        
        Important:
        - Ensure the response is valid JSON
        - Keep the summary concise but informative
        - Include all relevant sources
        - List all tools used
        - Do not include any text outside the JSON object
        """
        
        # Get response from LLM
        response = llm.invoke(prompt)
        
        # Clean and parse the response
        response_dict = clean_json_response(response)
        
        # Create ResearchResponse object
        return ResearchResponse(
            topic=response_dict.get("topic", "Research Topic"),
            summary=response_dict.get("summary", "No summary available"),
            sources=response_dict.get("sources", ["Information from search results"]),
            tools_used=response_dict.get("tools_used", ["Wikipedia", "Web Search"])
        )
        
    except Exception as e:
        print(f"Error during research: {e}")
        return None

def main():
    print("Welcome to the Research Assistant!")
    print("This tool will help you research topics using Wikipedia and web search.")
    
    while True:
        query = input("\nWhat would you like to research? (or type 'quit' to exit): ")
        
        if query.lower() == 'quit':
            break
            
        print("\nResearching... This may take a few moments.")
        result = research_topic(query)
        
        if result:
            print("\nResearch Results:")
            print(f"Topic: {result.topic}")
            print(f"\nSummary: {result.summary}")
            print("\nSources:")
            for source in result.sources:
                print(f"- {source}")
            print("\nTools Used:")
            for tool in result.tools_used:
                print(f"- {tool}")
            
            # Save the results
            save_tool.run(str(result))
            print("\nResults have been saved to research_output.txt")
        else:
            print("\nSorry, I couldn't complete the research. Please try again with a different query.")

if __name__ == "__main__":
    main()