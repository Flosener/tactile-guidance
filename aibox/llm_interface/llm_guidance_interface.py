import asyncio
import threading
from queue import Queue
from typing import Optional
from llm_interface import llm_interface_client

class LLMGuidanceInterface:
    def __init__(self):
        self.input_queue = Queue()
        self.output_queue = Queue() # queue for commands to be sent back to the system
        self.llm_client: Optional[llm_interface_client.LLMInterfaceClient] = None
        self.running = False
        self.input_thread: Optional[threading.Thread] = None
        self.llm_thread: Optional[threading.Thread] = None

    def start(self):
        """Start the interface threads"""
        self.running = True
        self.input_thread = threading.Thread(target=self._input_loop)
        self.llm_thread = threading.Thread(target=self._llm_loop)
        self.input_thread.start()
        self.llm_thread.start()

    def stop(self):
        """Stop the interface threads"""
        self.running = False
        if self.input_thread:
            self.input_thread.join()
        if self.llm_thread:
            self.llm_thread.join()

    def _input_loop(self):
        """Listen for console input"""
        while self.running:
            try:
                user_input = input()
                self.input_queue.put(user_input)
            except EOFError:
                continue

    def _llm_loop(self):
        """Process input and communicate with LLM interface"""
        async def run_llm():
            async with llm_interface_client.LLMInterfaceClient() as client:
                self.llm_client = client
                while self.running:
                    if not self.input_queue.empty():
                        user_input = self.input_queue.get()
                        try:
                            response = await client.invoke(user_input)
                            command = self._handle_llm_response(response)
                            print(f"Interpreted command: {command!r}")
                            if command:
                                print(f"Sending command to system: {command!r}")
                                self.output_queue.put(command)
                        except Exception as e:
                            print(f"Error communicating with LLM interface: {e}")
                    await asyncio.sleep(0.1)

        asyncio.run(run_llm())

    def _handle_llm_response(self, response: dict):
        """Get the command key from LLM response"""
        try:
            # Get the last AI message content
            messages = response.get("output", {}).get("messages", [])
            print(f"Processing response with {len(messages)} messages")
            ai_messages = [m for m in messages if m.get("type") == "ai"]
            print(f"Found {len(ai_messages)} AI messages")
            
            if not ai_messages:
                print("No AI messages found in response")
                return None
                
            last_message = ai_messages[-1]
            content = last_message.get("content", "").strip()
            print(f"Raw command content: {content!r}")
            
            # Check if content is a valid command
            valid_commands = {'s', 'y', 'n', 'f', 't', 'c'}
            
            # Check if content is a valid object ID
            try:
                object_id = int(content)
                if object_id in {1, 39, 40, 41, 42, 45, 46, 47, 58, 74}:
                    return content # return object ID as a string
            except ValueError:
                pass # not an object ID, continue checking other commands
            # After checking for object ID, check for standard commands
            if content in valid_commands:
                return content
            else:
                print("Not a valid command or object ID, ignoring")
                return None
            
        except Exception as e:
            print(f"Error handling LLM response: {e}")
            return None