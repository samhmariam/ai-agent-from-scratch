import marimo

__generated_with = "0.23.2"
app = marimo.App(width="medium")


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    Chapter 3: Tools and Function Calling
    """)
    return


@app.cell
def _():
    import marimo as mo

    return (mo,)


@app.cell
def _():
    from dotenv import load_dotenv, find_dotenv

    load_dotenv(find_dotenv())
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    3.1 Tool Definition Schema
    """)
    return


@app.cell
def _():
    import json

    # Define a calculator tool manually using the OpenAI function calling format
    calculator_tool_definition = {
        "type": "function",
        "function": {
            "name": "calculator",
            "description": "Perform basic arithmetic operations.",
            "parameters": {
                "type": "object",
                "properties": {
                    "operator": {
                        "type": "string",
                        "description": "Arithmetic operation to perform",
                        "enum": ["add", "subtract", "multiply", "divide"],
                    },
                    "first_number": {
                        "type": "number",
                        "description": "First number for the calculation",
                    },
                    "second_number": {
                        "type": "number",
                        "description": "Second number for the calculation",
                    },
                },
                "required": ["operator", "first_number", "second_number"],
            },
        },
    }
    return calculator_tool_definition, json


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    3.2 Automated Tool Definition
    """)
    return


@app.function
def calculator(operator: str, first_number: float, second_number: float):
    if operator == "add":
        return first_number + second_number
    elif operator == "subtract":
        return first_number - second_number
    elif operator == "multiply":
        return first_number * second_number
    elif operator == "divide":
        if second_number == 0:
            raise ValueError("Cannot divide by zero")
        return first_number / second_number
    else:
        raise ValueError(f"Unsupported operator: {operator}")


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    3.3 Tool Execution
    """)
    return


@app.cell
def _():
    from litellm import completion

    return (completion,)


@app.cell
def _(calculator_tool_definition, completion):
    tools = [calculator_tool_definition]

    response_without_tool = completion(
        model="gpt-5-mini",
        messages=[
            {"role": "user", "content": "What is the capital of South Korea?"}
        ],
        tools=tools,
    )

    print(response_without_tool.choices[0].message.content)
    print(response_without_tool.choices[0].message.tool_calls)

    messages = [{"role": "user", "content": "What is 1234 * 5678?"}]

    response_with_tool = completion(
        model="gpt-5-mini",
        messages=messages,
        tools=tools,
    )

    print(response_with_tool.choices[0].message.content)
    print(response_with_tool.choices[0].message.tool_calls)
    return messages, response_with_tool


@app.cell
def _(completion, json, messages, response_with_tool):
    ai_message = response_with_tool.choices[0].message

    messages.append({ 
        "role": "assistant", 
        "content": ai_message.content, 
        "tool_calls": ai_message.tool_calls 
    })

    if ai_message.tool_calls:
        for tool_call in ai_message.tool_calls:
            function_name = tool_call.function.name 
            function_args = json.loads(tool_call.function.arguments) 

            if function_name == "calculator":
                result = calculator(**function_args) 

    messages.append({ 
        "role": "tool", 
        "tool_call_id": tool_call.id, 
        "content": str(result) 
    })

    final_response = completion(
        model='gpt-5-mini',
        messages=messages
    )

    print("Messages: ", messages)
    print("Final Answer:", final_response.choices[0].message.content)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    3.4 Web Search Tool
    """)
    return


@app.cell
def _():
    import os
    from tavily import TavilyClient

    tavily_client = TavilyClient(os.getenv("TAVILY_API_KEY"))

    def search_web(
        query: str,
        max_results: int = 5,
        topic: str = "general",
        time_range: str | None = None,
    ) -> list | str:
        try:
            response = tavily_client.search(
                query,
                max_results=max_results,
                topic=topic,
                time_range=time_range,
            )
            return response.get("results")
        
        except Exception as e:
            return f"Error: Search failed - {e}"



    return os, search_web, tavily_client


@app.cell
def _(search_web):
    search_web("Kipchoge's marathon world record")
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    Converting to tool definitions
    """)
    return


@app.cell
def _():
    import inspect

    def function_to_input_schema(func) -> dict:
        type_map = {
            str: "string",
            int: "integer",
            float: "number",
            bool: "boolean",
            list: "array",
            dict: "object",
            type(None): "null",
        }
    
        try:
            signature = inspect.signature(func)
        except ValueError as e:
            raise ValueError(
                f"Failed to get signature for function {func.__name__}: {str(e)}"
            )
        
        parameters = {}
        for param in signature.parameters.values():
            try:
                param_type = type_map.get(param.annotation, "string")
            except KeyError as e:
                raise KeyError(
                f"Unknown type annotation {param.annotation} for parameter {param.name}: {str(e)}"
            )
            
            parameters[param.name] = {"type": param_type}
        
        required = [
            param.name
            for param in signature.parameters.values()
            if param.default == inspect._empty
        ]
    
        return {
            "type": "object",
            "properties": parameters,
            "required": required,
        }

    return (function_to_input_schema,)


@app.cell
def _(function_to_input_schema, search_web):
    function_to_input_schema(search_web) 
    return


@app.cell
def _(function_to_input_schema):
    def format_tool_definition(name: str, description: str, parameters: dict) -> dict:
        return {
            "type": "function",
            "function": {
                "name": name,
                "description": description,
                "parameters": parameters,
            },
        }
    
    def function_to_tool_definition(func) -> dict:
        return format_tool_definition(
            func.__name__,
            func.__doc__ or "",
            function_to_input_schema(func)
        )

    return format_tool_definition, function_to_tool_definition


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    BUILDING THE TOOL EXECUTION SYSTEM
    """)
    return


@app.cell
def _(json):
    def tool_execution(tool_box, tool_call):
        function_name = tool_call.function.name
        function_args = json.loads(tool_call.function.arguments)
    
        tool_result = tool_box[function_name](**function_args)
        return tool_result


    return (tool_execution,)


@app.cell
def _(completion, function_to_tool_definition, search_web, tool_execution):
    def simple_agent_loop(system_prompt, question):
        tools = [search_web]
        tool_box = {tool.__name__: tool for tool in tools}
        tool_definitions = [function_to_tool_definition
                            (tool) for tool in tools]
    
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": question}
        ]
    
        while True:
            response = completion(
                model="gpt-5-mini",
                messages=messages,
                tools=tool_definitions
            )
        
            assistant_message = response.choices[0].message
            if assistant_message.tool_calls:
                messages.append(assistant_message)
            
                for tool_call in assistant_message.tool_calls:
                    tool_result = tool_execution(tool_box, tool_call)
                    messages.append({
                        "role": "tool",
                        "content": str(tool_result),
                        "tool_call_id": tool_call.id
                    })
            else:
                return assistant_message.content

    return (simple_agent_loop,)


@app.cell
def _(simple_agent_loop):
    system_prompt = """You are a helpful assistant.
        Use the search tool when you need current information."""

    _result = simple_agent_loop(
        system_prompt,
        "Who won the 2025 Nobel Prize in Physics?"
    )
    print(_result)

    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    3.5 MCP Integration
    """)
    return


@app.cell
async def _(os):
    import asyncio
    from mcp import ClientSession, StdioServerParameters
    from mcp.client.stdio import stdio_client

    server_params = StdioServerParameters(
        command="npx",
        args=["-y", "tavily-mcp@latest"],
        env={
            "TAVILY_API_KEY": os.getenv("TAVILY_API_KEY"),
        }
    )

    async with stdio_client(server_params) as (read_stream, write_stream):
        async with ClientSession(read_stream, write_stream) as session:
            await session.initialize()
        
            # List available tools
            tools_result = await session.list_tools()
            print("Available tools:")
        
            for tool in tools_result.tools:
                print(f" - {tool.name}: {tool.description[:60]}...")
            
            _result = await session.call_tool(
                "tavily_search",
                arguments={"query": "2025 Nobel Physics"}
            )
    print("Search Result:")
    print(_result.content)

    return (asyncio,)


@app.cell
def _(format_tool_definition):
    def mcp_tools_to_openai_format(mcp_tools) -> list[dict]:
        """Convert MCP tool definitions to OpenAI tool format."""
        return [
            format_tool_definition(
                name=tool.name,
                description=tool.description,
                parameters=tool.inputSchema,
            )
            for tool in mcp_tools.tools
        ]

    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    BUILDING THE SERVER
    """)
    return


@app.cell
def _(asyncio, tavily_client):
    from mcp.server.fastmcp import FastMCP

    mcp = FastMCP("custom-tavily-search")

    @mcp.tool()
    def _search_web(query: str, max_results: int = 5) -> str:
        """
        Search the web using Tavily API.
    
        Args:
            query: Search query string
            max_results: Maximum number of results to return (default: 5)
    
        Returns:
            Search results as formatted string
        """
        try:
            response = tavily_client.search(
                query,
                max_results=max_results,
            )
            results = response.get("results", [])
            return "\n\n".join(
                f"Title: {r['title']}\nURL: {r['url']}\nContent: {r['content']}"
                for r in results
            )
        except Exception as e:
            return f"Error searching web: {str(e)}"
    if __name__ == "__main__":
        # If an asyncio event loop is already running (e.g., inside an async environment),
        # avoid calling mcp.run() directly to prevent "RuntimeError: Already running asyncio in this thread".
        try:
            asyncio.get_running_loop()
        except RuntimeError:
            mcp.run(transport='stdio')
        else:
            # Run MCP in a separate thread to avoid blocking the existing event loop
            import threading
            def _run_mcp():
                mcp.run(transport='stdio')
            t = threading.Thread(target=_run_mcp, daemon=True)
            t.start()
    return


if __name__ == "__main__":
    app.run()
