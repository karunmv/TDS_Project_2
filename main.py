# main.py

# --- 1. Import necessary libraries ---
import os
import base64
import traceback
import io
import logging
import json
import uuid
from contextlib import redirect_stdout
from fastapi import FastAPI, UploadFile, File, HTTPException
from typing import List
from openai import OpenAI

# --- 2. Basic Configuration ---
logging.basicConfig(level=logging.INFO)

# IMPORTANT: Set your OpenAI API key as an environment variable.
# For local testing, you can create a .env file or temporarily set it here like this:
# os.environ["OPENAI_API_KEY"] = "sk-YourSecretKeyGoesHere"
client = OpenAI()
app = FastAPI()

# Create a directory to temporarily store files for each request
UPLOAD_DIRECTORY = "./uploads"
if not os.path.exists(UPLOAD_DIRECTORY):
    os.makedirs(UPLOAD_DIRECTORY)

# --- 3. The System Prompt: This is the agent's "instructions" ---
# This detailed prompt tells the LLM exactly how to behave and what kind of output we need.
SYSTEM_PROMPT = """
You are a world-class data analyst agent. Your goal is to help users by answering their questions about data.
You MUST respond with a single, executable Python 3 script. Do not provide any explanation.

Your Python script should:
1. Import any necessary libraries. The following are pre-installed: pandas, numpy, matplotlib, seaborn, scikit-learn, requests, beautifulsoup4.
2. Load data from the provided files. The files are in the same directory your script will run in.
3. Perform all the necessary analysis to answer the user's questions.
4. For any plots, generate the plot, save it to a BytesIO buffer, and encode it as a base64 data URI string (`data:image/png;base64,...`). The final string MUST be under 100,000 bytes. DO NOT save plots to a file.
5. The final line of your script MUST be a single `print()` statement containing the complete JSON result as a string.

Example of handling a plot:
```python
import matplotlib.pyplot as plt
import io
import base64
import json
# ... (your plotting code) ...
buf = io.BytesIO()
plt.savefig(buf, format='png', bbox_inches='tight')
plt.close()
plot_base64 = base64.b64encode(buf.read()).decode('utf-8')
data_uri = f"data:image/png;base64,{plot_base64}"
# ... add data_uri to your final result dict/list ...
# print(json.dumps(final_answers))
"""

def execute_python_code(code: str, session_dir: str) -> str:
    """
    Executes the LLM-generated Python code safely in a specific directory
    and captures its standard output.
    """
    output_buffer = io.StringIO()
    original_cwd = os.getcwd()
    try:
        # Change to the session directory so the script can find uploaded files
        os.chdir(session_dir)
        with redirect_stdout(output_buffer):
        # IMPORTANT: In a real-world production app, this exec call MUST be sandboxed
        # for security using Docker or a similar technology. For this project, it's okay.
        exec(code, globals())

        result = output_buffer.getvalue().strip()
        if not result:
            raise ValueError("The executed script produced no output. It must end with a print() statement.")
        return result
    except Exception as e:
        # Log the full error and return a JSON error message
        logging.error(f"Code execution failed: {e}\n{traceback.format_exc()}")
        return json.dumps({"error": str(e), "traceback": traceback.format_exc()})
    finally:
        # Always change back to the original directory
        os.chdir(original_cwd)

@app.post("/api/")
async def analyze_data(files: List[UploadFile] = File(...)):
    """
    This endpoint handles the incoming request with files.
    """
    if not files:
    raise HTTPException(status_code=400, detail="No files were sent.")

    # Create a unique temporary directory for this specific request
    session_id = str(uuid.uuid4())
    session_dir = os.path.join(UPLOAD_DIRECTORY, session_id)
    os.makedirs(session_dir)

    questions = ""
    file_names = []

    # Save all uploaded files into the unique directory
    for file in files:
        file_path = os.path.join(session_dir, file.filename)
        with open(file_path, "wb") as buffer:
            buffer.write(await file.read())

        if file.filename == "questions.txt":
            with open(file_path, "r") as f:
                questions = f.read()
        else:
            file_names.append(file.filename)

    if not questions:
        raise HTTPException(status_code=400, detail="A 'questions.txt' file is required.")

    # --- 6. Construct the Prompt for the LLM ---
    user_prompt = f"Here are the questions:\n{questions}\n"
    if file_names:
        user_prompt += f"\nUse the following data file(s): {', '.join(file_names)}"
    else:
        user_prompt += f"\nYou will need to source the data yourself (e.g., by scraping a website)."

    logging.info("Sending prompt to LLM...")

    # --- 7. Call the LLM to Generate Code ---
    try:
        response = client.chat.completions.create(
            model="gpt-4o",
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": user_prompt}
            ],
            temperature=0.0
        )
        generated_code = response.choices[0].message.content.strip()

        # Clean up the response in case the LLM wraps it in markdown
        if generated_code.startswith("```python"):
            generated_code = generated_code[9:]
        if generated_code.endswith("```"):
            generated_code = generated_code[:-3]

        logging.info(f"LLM Generated Code:\n---\n{generated_code}\n---")

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error calling LLM: {e}")

    # --- 8. Execute the Generated Code ---
    logging.info("Executing generated code...")
    result_json_str = execute_python_code(generated_code, session_dir)

    # --- 9. Clean Up ---
    for file_name in os.listdir(session_dir):
        os.remove(os.path.join(session_dir, file_name))
    os.rmdir(session_dir)

    # --- 10. Return the Result ---
    # The result is already a JSON string, so we return it directly.
    # The real response will have the correct `application/json` content type.
    return json.loads(result_json_str)

@app.get("/")
def read_root():
return {"message": "Data Analyst Agent API is active."}