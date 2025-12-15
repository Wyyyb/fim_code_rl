import os
import logging
from google import genai

# Configure logging
logger = logging.getLogger(__name__)


class GeminiClientWithoutSearch:
    def __init__(self, model: str = "gemini-2.5-flash"):
        api_key = os.getenv("GOOGLE_API_KEY")
        if not api_key:
            raise ValueError("GOOGLE_API_KEY environment variable not set")
        os.environ["GOOGLE_API_KEY"] = api_key
        self.client = genai.Client()
        self.model = model

    def get_response(self, prompt: str) -> str:
        """
        Get response from Gemini API without web search.

        Args:
            prompt: The input prompt
        """
        temperature = 0.7
        logger.info(f"🔄 Calling Gemini API (no search)...")
        logger.info(f"  Model: {self.model}")
        logger.info(f"  Temperature: {temperature}")
        logger.info(f"  Prompt length: {len(prompt)} characters")

        try:
            response = self.client.models.generate_content(
                model=self.model,
                contents=prompt,
                config={
                    "temperature": temperature,
                    "top_p": 0.9,
                },
            )
            logger.info(f"✅ Gemini API call successful")
            logger.info(f"  Response length: {len(response.text)} characters")
            return response.text
        except Exception as e:
            logger.error(f"❌ Gemini API call failed: {str(e)}")
            raise


class GeminiClientWithSearch:
    def __init__(self, model: str = "gemini-2.5-pro"):
        api_key = os.getenv("GOOGLE_API_KEY")
        if not api_key:
            raise ValueError("GOOGLE_API_KEY environment variable not set")
        os.environ["GOOGLE_API_KEY"] = api_key
        self.client = genai.Client()
        self.model = model

    def get_response(self, prompt: str) -> str:
        """
        Get response from Gemini API with web search grounding enabled.

        Args:
            prompt: The input prompt
        """
        temperature = 0.7
        logger.info(f"🔄 Calling Gemini API (with search)...")
        logger.info(f"  Model: {self.model}")
        logger.info(f"  Temperature: {temperature}")
        logger.info(f"  Prompt length: {len(prompt)} characters")

        try:
            response = self.client.models.generate_content(
                model=self.model,
                contents=prompt,
                config={
                    "tools": [{"google_search": {}}],
                    "temperature": temperature,
                    "top_p": 0.9,
                },
            )
            logger.info(f"✅ Gemini API call successful")
            logger.info(f"  Response length: {len(response.text)} characters")
            return response.text
        except Exception as e:
            logger.error(f"❌ Gemini API call failed: {str(e)}")
            raise



def test_flash():
    prompt = "hello, gemini!"
    client = GeminiClientWithoutSearch(model="gemini-2.5-flash")
    response = client.get_response(prompt)
    print("response:", response)


if __name__ == "__main__":
    test_flash()



