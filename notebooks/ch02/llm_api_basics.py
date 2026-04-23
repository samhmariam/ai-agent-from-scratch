import marimo

__generated_with = "0.23.2"
app = marimo.App(width="medium")


@app.cell
def _():
    import marimo as mo

    return (mo,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    Chapter 2: LLM API Basics
    """)
    return


@app.cell
def _():
    from dotenv import load_dotenv, find_dotenv
    load_dotenv(find_dotenv())

    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    2.1 OpenAI API
    """)
    return


@app.cell
def _():
    from openai import OpenAI

    client = OpenAI()

    response = client.chat.completions.create(
        model="gpt-5-mini",
        messages=[
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": "What is the capital of France?"}
        ]
    )
    print(response.choices[0].message.content)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    2.2 LiteLLM - Unified API
    """)
    return


@app.cell
def _():
    from litellm import completion

    # Call OpenAI via LiteLLM
    response_openai = completion(
        model="gpt-5-mini",
        messages=[{"role": "user", "content": "What is the capital of France?"}]
    )
    print("OpenAI:", response_openai.choices[0].message.content)

    return (completion,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    2.3 Conversation History
    """)
    return


@app.cell
def _(completion):

    messages = []

    # First exchange
    messages.append({"role": "user", "content": "My name is London."})
    response1 = completion(model="gpt-5-mini", messages=messages)
    assistant_message1 = response1.choices[0].message.content
    messages.append({"role": "assistant", "content": assistant_message1})

    print(assistant_message1)

    # Second exchange - includes previous conversation history
    messages.append({"role": "user", "content": "What is my name?"})
    response2 = completion(model="gpt-5-mini", messages=messages)
    assistant_message2 = response2.choices[0].message.content

    print(assistant_message2)

    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    2.4 Structured Output
    """)
    return


@app.cell
def _(completion):
    from pydantic import BaseModel

    class ExtractedInfo(BaseModel):
        name: str
        email: str
        phone: str | None = None
    
    _response = completion(
        model="gpt-5-mini",
        messages=[{
            "role": "user",
            "content": "My name is John Smith, my email is john@example.com, and my phone is 555-1234."
        }],
        response_format=ExtractedInfo
    )
    result = _response.choices[0].message.content

    print(result)
    return (BaseModel,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    2.5 Async Calls
    """)
    return


@app.cell
async def _():
    import asyncio
    from litellm import acompletion

    async def get_response(prompt: str) -> str:
        response = await acompletion(
            model="gpt-5-mini",
            messages=[{"role": "user", "content": prompt}]
        )
        return response.choices[0].message.content

    prompts = [
        "What is 2 + 2?",
        "What is the capital of Japan?",
        "Who wrote Romeo and Juliet?"
    ]

    # Execute all requests concurrently
    tasks = [get_response(p) for p in prompts]
    _results = await asyncio.gather(*tasks)

    for prompt, _result in zip(prompts, _results):
        print(f"Q: {prompt}")
        print(f"A: {_result}\n")

    return acompletion, asyncio


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    2.6 GAIA Benchmark Evaluation
    """)
    return


@app.cell
def _():
    from datasets import load_dataset

    level1_problems = load_dataset("gaia-benchmark/GAIA", "2023_level1", split="validation")

    print(f"Number of Level 1 problems: {len(level1_problems)}")
    return (level1_problems,)


@app.cell
def _(BaseModel):
    class GaiaOutput(BaseModel):
        is_solvable: bool
        unsolvable_reason: str = ""
        final_answer: str = ""


    return (GaiaOutput,)


@app.cell
def _():
    gaia_prompt="""You are a general AI assistant. I will ask you a question.
        First, determine if you can solve this problem with your current capabilities and
        set "is_solvable" accordingly.
        If you can solve it, set "is_solvable" to true and provide your answer in
        "final_answer".
        If you cannot solve it, set "is_solvable" to false and explain why in
        "unsolvable_reason".
        Your final answer should be a number OR as few words as possible OR a comma
        separated list of numbers and/or strings.
        If you are asked for a number, don't use comma to write your number neither
        use units such as $ or percent sign unless specified otherwise.
        If you are asked for a string, don't use articles, neither abbreviations (e.g. for
        cities), and write the digits in plain text unless specified otherwise.
        If you are asked for a comma separated list, apply the above rules depending on
        whether the element is a number or a string."""
    return (gaia_prompt,)


@app.cell
def _(asyncio):
    PROVIDER_SEMAPHORES = {
        "openai": asyncio.Semaphore(10),
        "anthropic": asyncio.Semaphore(10),
    }

    def get_provider(model: str) -> str:
        """Extract provider name from model string."""
        return "anthropic" if model.startswith("anthropic/") else "openai"


    return PROVIDER_SEMAPHORES, get_provider


@app.cell
def _(GaiaOutput, PROVIDER_SEMAPHORES, acompletion, gaia_prompt, get_provider):
    async def solve_problem(model: str, question: str) -> GaiaOutput:
        """Solve a single problem and return structured output."""
        provider = get_provider(model)
        async with PROVIDER_SEMAPHORES[provider]:
            response = await acompletion(
                model=model,
                messages=[
                    {"role": "system", "content": gaia_prompt},
                    {"role": "user", "content": question},
                ],
                response_format=GaiaOutput,
                num_retries=2,
            )
            finish_reason = response.choices[0].finish_reason
            content = response.choices[0].message.content
            if finish_reason == "refusal" or content is None:
                return GaiaOutput(
                    is_solvable=False,
                    unsolvable_reason=f"Model refused to answer (finish_reason: {finish_reason})",
                    final_answer=""
                )
            return GaiaOutput.model_validate_json(content)


    return (solve_problem,)


@app.function
def is_correct(prediction: str | None, answer: str) -> bool:
    """Check exact match between prediction and answer (case-insensitive)."""
    if prediction is None:
        return False
    return prediction.strip().lower() == answer.strip().lower()


@app.cell
def _(solve_problem):
    async def evaluate_gaia_single(problem: dict, model: str) -> dict:
        """Evaluate a single problem-model pair and return result."""
        try:
            output = await solve_problem(model, problem["Question"])
            return {
                "task_id": problem["task_id"],
                "model": model,
                "correct": is_correct(output.final_answer, problem["Final answer"]),
                "is_solvable": output.is_solvable,
                "prediction": output.final_answer,
                "answer": problem["Final answer"],
                "unsolvable_reason": output.unsolvable_reason,
            }
        except Exception as e:
            return {
                "task_id": problem["task_id"],
                "model": model,
                "correct": False,
                "is_solvable": None,
                "prediction": None,
                "answer": problem["Final answer"],
                "error": str(e),
            }

    return (evaluate_gaia_single,)


@app.cell
def _(evaluate_gaia_single):
    from tqdm.asyncio import tqdm_asyncio

    async def run_experiment(problems: list[dict], models: list[str],) -> dict[str, list]:
        """Evaluate all models on all problems."""
    
        tasks = [
            evaluate_gaia_single(problem, model)
            for problem in problems
            for model in models
        ]
        all_results = await tqdm_asyncio.gather(*tasks)
    
        # Group results by model
        results = {model: [] for model in models}
        for result in all_results:
            results[result["model"]].append(result)
        
        return results


    return (run_experiment,)


@app.cell
async def _(level1_problems, run_experiment):
    MODELS = [
        "gpt-5",
        "gpt-5-mini",
    ]
    subset = level1_problems.select(range(5))
    results = await run_experiment(subset, MODELS)

    return (results,)


@app.cell
def _(results):
    results
    return


if __name__ == "__main__":
    app.run()
