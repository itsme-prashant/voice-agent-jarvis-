import os
from dotenv import load_dotenv

# Ensure we're loading from the same directory as the project root
base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
env_path = os.path.join(base_dir, '.env')
load_dotenv(dotenv_path=env_path)

class Settings:
    OPENAI_API_KEY: str = os.getenv("OPENAI_API_KEY", "")
    OPENWEATHER_API_KEY: str = os.getenv("OPENWEATHER_API_KEY", "")
    GOOGLE_CALENDAR_API_KEY: str = os.getenv("GOOGLE_CALENDAR_API_KEY", "")
    
    # Other future config
    MCP_SERVERS = []

settings = Settings()
