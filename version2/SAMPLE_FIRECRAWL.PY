from phi.agent import Agent
from firecrawl import FirecrawlApp
import os
from dotenv import load_dotenv
load_dotenv()

# Initialize Firecrawl directly
firecrawl = FirecrawlApp(api_key=os.getenv("FIRECRAWL_API_KEY"))

# Create a custom scraping function
def scrape_website_custom(url: str):
    try:
        # Try the new API method
        result = firecrawl.scrape(url=url)
        return result
    except AttributeError:
        try:
            # Try alternative method
            result = firecrawl.scrape_url(url)
            return result
        except:
            return {"error": "Could not scrape website"}

# Use Agent without FirecrawlTools
agent = Agent(
    show_tool_calls=True,
    markdown=True
)

# Manually scrape and then ask agent to summarize
url = "https://in.indeed.com/viewjob?jk=4efedf0479edfd1c&l=India&tk=1j9eeeefdhbm186m&from=web&advn=9583604388004447&adid=453015361&ad=-6NYlbfkN0BOlSn2mBENrgA-Nx4GbtBE-S7LCZREBx_YojhWDYVvufHn0aQV8ukB2bOjX6y5KW2hQFvlR3SEYaMQRwiyiYisj-E_Gbzfwn3CmYvwN7bXcthHj4IBANB6wRqB3WGFz78xACxzGTBidb05-FkLS2FSKJ6IM0VeVWgHplaq_zSU1H-ocJI1YgEHa4IvDHlIG3jo6iYlhH9yjKyqNuk7TAzIldBMy6fvcVdQH-9p8BWB1LWuyi_jxcyPvsjZP4Jnfacat1Ow4E-2Bc95w2GX6Dj8It5SW1xN-Kx3eWRC48JDclaeileo5Aj8Yi0Qg4RVmQpPmXPQjagZREA8luzVz8gZUE9T8q3YWo94Fp9wHCSiWEmVKuxfFGdDh_bx3pmQdbhX2E9cASuk3_oaapUpr0bNOmnthuEaLe0YEGkld1ew7r9dOoYPAk_Rfj_-fuQDVYTPccYrvqTDX7s4WVXJcYd6Q7tKb7UeGtnPOkvC4mosUs8wagztswBf9obN8vt2lPqb3PPb01qLuj0Ws6uP-gb5bKFOh8m4fB2S31zpJkUI9sXJ3QWO0DWW&pub=4a1b367933fd867b19b072952f68dceb&camk=UoKtGZLa3XKfc8RDgMWGhg%3D%3D&xkcb=SoBX6_M3ruU2AmTNzJ0LbzkdCdPP&xpse=SoAH6_I3ruU1HBzF7Z0IbzkdCdPP&xfps=579d4ddc-f07e-4117-aa06-604c6de48492&vjs=3"
scraped_data = scrape_website_custom(url)

if "error" not in scraped_data:
    agent.print_response(f"Summarize this content: {scraped_data}")
else:
    print("Error scraping website")