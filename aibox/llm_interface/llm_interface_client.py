import aiohttp
import json
from typing import Optional, Dict, Any
from labels import coco_labels

class LLMInterfaceClient:
    """
    Client for communicating with the Context-Aware LLM Navigation Interface for Accessibility Apps.
    Handles authentication, message passing, and response processing.
    https://github.com/RillJ/llm-app-interface
    """
    def __init__(self, base_url: str = "http://localhost:8000", username: str = "johndoe", password: str = "secret"):
        """
        Initialize the LLM interface client.
        
        Args:
            base_url: URL of the LLM API server
            username: Authentication username
            password: Authentication password
        """
        self.base_url = base_url
        self.username = username
        self.password = password
        self.token: Optional[str] = None # JWT token for authentication
        self.session: Optional[aiohttp.ClientSession] = None # HTTP session for requests

    async def __aenter__(self):
        self.session = aiohttp.ClientSession()
        await self.authenticate()
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        if self.session:
            await self.session.close()

    async def authenticate(self):
        """Get JWT token for API authentication"""
        if not self.session:
            return

        async with self.session.post(
            f"{self.base_url}/token",
            data={"username": self.username, "password": self.password}
        ) as response:
            if response.status == 200:
                data = await response.json()
                self.token = data["access_token"]
            else:
                raise Exception("Authentication failed")

    def _get_additional_data(self, label: str) -> Dict[str, Any]:
        """Get additional data based on the function label"""
        if label == "object-and-hand-recognition":
            valid_object_ids = [1, 39, 40, 41, 42, 45, 46, 47, 58, 74] # Same as classes_obj in master.py

            return {
                "available_objects": [
                    {"id": str(id), "name": coco_labels[id], "description": f"A {coco_labels[id]} that can be {'grasped' if id in [39, 41, 45] else 'detected'}"} 
                    for id in valid_object_ids
                ]
            }
        elif label == "command-interface":
            return {
                "available_commands": [
                    {"command": "s", "description": "Start a new trial"},
                    {"command": "y", "description": "Confirm successful grasp"},
                    {"command": "n", "description": "Indicate failed grasp"},
                    {"command": "t", "description": "Wrong target was grasped"},
                    {"command": "f", "description": "System failure occurred"},
                    {"command": "q", "description": "Quit the system"},
                    {"command": "c", "description": "Cancel current trial"}
                ]
            }
        return {}

    async def invoke(self, message: str) -> Dict[Any, Any]:
        """
        Send a message to the LLM API and handle the response flow.
        
        This method implements a two-step communication process:
        1. First request to get function metadata and check if additional data is needed
        2. Second request (if needed) to send additional context data to the LLM
        
        Args:
            message: The user's input message to process
            
        Returns:
            Dict containing the LLM's response with messages and any additional data
            
        Raises:
            Exception: If not authenticated or if API calls fail
        """
        if not self.session or not self.token:
            raise Exception("Not authenticated")

        # Set up authentication and content type headers
        headers = {
            "Authorization": f"Bearer {self.token}",
            "Content-Type": "application/json"
        }

        # Prepare initial request body with user message
        request_body = {
            "input": {
                "messages": message
            }
        }
        
        # Get the function metadata and determine if we need additional data
        async with self.session.post(
            f"{self.base_url}/llm-app-interface/invoke",
            headers=headers,
            json=request_body
        ) as response:
            if response.status != 200:
                error_text = await response.text()
                raise Exception(f"API call failed with status {response.status}: {error_text}")
            
            # Parse the metadata response
            metadata = await response.json()
            # Response format is "label=X; additional-data-required=Y"
            if isinstance(metadata, dict) and "output" in metadata:
                content = metadata["output"].get("messages", [])[-1].get("content", "")
                parts = content.split(";")
                label = "" # Function label
                needs_data = False # Flag indicating if additional context is needed
                
                for part in parts:
                    if "label=" in part:
                        label = part.split("=")[1].strip()
                    elif "additional-data-required=" in part:
                        needs_data = part.split("=")[1].strip().lower() == "true"

            # If additional context is needed, prepare and send it
            if needs_data:
                print(f"\nLLM requires additional data for {label}")
                # Get the appropriate additional data based on the function label
                additional_data = self._get_additional_data(label)
                # Format the data with <App> prefix as expected by the LLM
                app_message = f"<App>{json.dumps(additional_data)}"
                # Prepare new request with the additional context
                request_body = {
                    "input": {
                        "messages": app_message
                    }
                }
                print(f"Sending additional data: {app_message}\n")
                
                # Make the request again with the additional data
                async with self.session.post(
                    f"{self.base_url}/llm-app-interface/invoke",
                    headers=headers,
                    json=request_body
                ) as response:
                    if response.status == 200:
                        response_data = await response.json()
                        # Get the last AI message content
                        if response_data.get("output", {}).get("messages"):
                            messages = response_data["output"]["messages"]
                            ai_messages = [m for m in messages if m.get("type") == "ai"]
                            if ai_messages:
                                last_message = ai_messages[-1]
                                print(f"\nLLM Response: {last_message.get('content')}\n")
                        return response_data
                    else:
                        error_text = await response.text()
                        raise Exception(f"API call failed with status {response.status}: {error_text}")
            else:
                # No additional data needed, just process the response
                if response.status == 200:
                    response_data = await response.json()
                    # Get the last AI message content
                    if response_data.get("output", {}).get("messages"):
                        messages = response_data["output"]["messages"]
                        ai_messages = [m for m in messages if m.get("type") == "ai"]
                        if ai_messages:
                            last_message = ai_messages[-1]
                            print(f"\nLLM Response: {last_message.get('content')}\n")
                    return response_data
                else:
                    error_text = await response.text()
                    raise Exception(f"API call failed with status {response.status}: {error_text}")