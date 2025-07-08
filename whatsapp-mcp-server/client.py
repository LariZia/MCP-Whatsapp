from mcp import ClientSession, StdioServerParameters, types
from mcp.client.stdio import stdio_client
import logging
import asyncio
import google.api_core.exceptions
from google import genai
from google.genai import types

from dotenv import load_dotenv
import os


load_dotenv()
client = genai.Client(
    api_key=os.getenv("GEMINI_API_KEY")
)

model = os.getenv("GEMINI_MODEL")

# functions declarations 

logging.basicConfig(level=logging.INFO)

server_params = StdioServerParameters(
                command="uv", 
                args=["run", "main.py"],
                env=None,
                )

############################
#   Helper fucntion to call gemini safely 
############################

async def generate_with_retry(client, model, contents, tools, max_retries=3, temperature=0.7):
    for attempt in range(max_retries):
        try:
            response = client.models.generate_content(
                model=model,
                contents=contents,
                config=types.GenerateContentConfig(
                    tools=tools,
                    temperature=temperature,
                ),
            )
            return response
        except genai.errors.ClientError as e:
            try:
                error_data = e.response.json()
            except Exception as json_err:
                print("Failed to parse error response:", json_err)
                raise e

            status = error_data.get("error", {}).get("status", "")
            if status == "RESOURCE_EXHAUSTED" or status == "429 RESOURCE_EXHAUSTED" or status == "Too Many Requests":
                retry_delay = 30  # fallback
                try:
                    for detail in error_data.get("error", {}).get("details", []):
                        if detail.get("@type") == "type.googleapis.com/google.rpc.RetryInfo":
                            retry_delay = int(detail.get("retryDelay", "30s").strip("s"))
                            break
                except Exception as parse_err:
                    print("Error parsing retry delay:", parse_err)

                if attempt < max_retries - 1:
                    print(f"🕒 Quota exceeded. Retrying in {retry_delay}s... (attempt {attempt+1}/{max_retries})")
                    await asyncio.sleep(retry_delay)
                else:
                    print("❌ Max retries hit. Exiting.")
                    raise e
            else:
                print("❌ Unexpected error:", error_data)
                raise e

# genai handles function calling itself
# Main async function
async def run():
    print("\n===== MCP CLIENT (Gemini) =====\n")

    async with stdio_client(server_params) as (read, write):
        async with ClientSession(read, write) as session:
            await session.initialize()
            print("session initialized")

             # Get tools from MCP session and convert to Gemini Tool objects
            mcp_tools = await session.list_tools()

            ## edited 
            # Convert tools to Gemini-style tool declarations
            tools = [
                {
                    "function_declarations": [
                        {
                            "name": tool.name,
                            "description": tool.description,
                            "parameters": tool.inputSchema,
                        }
                        for tool in mcp_tools.tools
                    ]
                }
            ]
            print(f"Available tools: {[tool.name for tool in mcp_tools.tools]}")

            # 🧠 Conversation memory across turns
            contents: list[types.Content] = []

            while True:
                user_input = input("\nYou: ").strip()
                if user_input.lower() == "exit":
                    print("Exiting...")
                    break

                # Add user input to conversation memory
                contents.append(
                    types.Content(role="user", parts=[types.Part(text=user_input)])
                )


                response = await generate_with_retry(client, model, contents, tools)
                        
                candidate = response.candidates[0]
                part = candidate.content.parts[0]
                print("###### PART ####", part)

                # Check for function call
                if hasattr(part, "function_call") and part.function_call is not None:
                    function_call = part.function_call
                    print(f"\n[Gemini requested tool: {function_call.name} with args {function_call.args}]")

                    try:
                        # Call the actual tool on MCP server
                        tool_result = await session.call_tool(function_call.name, arguments=function_call.args)
                        print("Tool result:", tool_result)


                        # Add model's function call and tool result to memory
                        contents.append(types.Content(
                            role="model",
                            parts=[types.Part(function_call=function_call)]
                        ))
                        contents.append(types.Content(
                            role="user",
                            parts=[
                                types.Part.from_function_response(
                                    name=function_call.name,
                                    response={"result": tool_result.content[0].text if tool_result.content else "No result returned"}
                                )
                            ]
                        ))

                        # Step 4: Send tool result back to Gemini for final user-friendly response
                        function_response_part = types.Part.from_function_response(
                            name=function_call.name,
                            response={"result": tool_result.content[0].text if tool_result.content else "No result returned"}
                        )

                        # Append both the model's call and the user-provided response
                        contents.append(types.Content(role="model", parts=[types.Part(function_call=function_call)]))
                        contents.append(types.Content(role="user", parts=[function_response_part]))

                        final_response = None

                        final_response = await generate_with_retry(client, model, contents, tools)

                        # Add final model response to memory
                        if final_response:
                            contents.append(final_response.candidates[0].content)
                            if final_response.candidates[0].content.parts:
                                for p in final_response.candidates[0].content.parts:
                                    if p.text:
                                        print(f"\nAssistant: {p.text}")

                    except Exception as e:
                        logging.error(f"Error calling tool {function_call.name}: {e}")
                        print(f"\nAssistant: Failed to execute tool: {function_call.name}")
                else:
                    # Fallback if no tool was called
                    # Safely extract text from candidate content
                    contents.append(candidate.content)
                    if candidate.content.parts:
                        for p in candidate.content.parts:
                            if p.text:
                                print(f"\nAssistant: {p.text}")
                            elif p.function_call:
                                print(f"\n[Gemini requested tool: {p.function_call.name} with args {p.function_call.args}]")
                    else:
                        print("\nAssistant: [No response from Gemini]")


if __name__ == "__main__":
    asyncio.run(run())