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
client = OpenAI(
    # The key is read from the environment variable you set on Render
    api_key=os.environ.get("OPENAI_API_KEY"), 

    # This tells the client to send requests to AIPipe instead of OpenAI
    base_url="https://aipipe.org/openai/v1" # <-- IMPORTANT: Replace with your actual AIPipe URL!
)
app = FastAPI()

# Create a directory to temporarily store files for each request
UPLOAD_DIRECTORY = "./uploads"
if not os.path.exists(UPLOAD_DIRECTORY):
    os.makedirs(UPLOAD_DIRECTORY)

# --- 3. The System Prompt: This is the agent's "instructions" ---
# This detailed prompt tells the LLM exactly how to behave and what kind of output we need.
SYSTEM_PROMPT = """
You are a world-class Data Analyst Agent.  
You MUST output a single, executable Python 3 script.  
Do not include explanations or markdown. Only valid Python code.

Your script must follow these rules:

1. **Imports**  
   - Allowed libraries: pandas, numpy, matplotlib, seaborn, scikit-learn, requests, beautifulsoup4, io, base64, json.  
   - Do not import anything else.  

2. **Data Loading**  
   - Input files are in the current working directory.  
   - Load them according to their type (CSV, JSON, Parquet, etc).  
   - If scraping is required, use `requests` + `BeautifulSoup`.  

3. **Column Handling**  
   - Always inspect `df.columns` before using.  
   - Do NOT hardcode column names; they may vary.  
   - Use case-insensitive or substring matching if appropriate (e.g., find `"gross"` in `"Worldwide gross"`).  
   - If a required column is missing, gracefully return JSON with `{"error": "...", "columns": [...]}` instead of crashing.  

4. **Data Types**  
   - Separate numeric and categorical columns using `df.select_dtypes()`.  
   - NEVER cast entire dataframes to float.  
   - If regression/ML is required, encode categorical variables with `pd.get_dummies(drop_first=True)`.  
   - Handle missing values with `.dropna()` or `.fillna()` appropriately.  

5. **Analysis**  
   - Only perform the operations required to answer the questions in `questions.txt`.  
   - Use appropriate methods for correlation, regression, aggregation, etc.  

6. **Plots**  
   - If a plot is required:  
     * Generate with matplotlib/seaborn.  
     * Save to a `BytesIO` buffer in PNG format.  
     * Base64 encode the bytes to `"data:image/png;base64,..."`
     * Ensure the string is **< 100,000 characters**.  
     * Always close figures after saving with `plt.close()`.  

7. **Output**  
   - If the user asks for an array of answers, output a JSON **list**.  
   - If the user asks for named answers, output a JSON **object**.  
   - Include plots as base64 URIs where needed.  
   - The **final line of the script must be:**  
     ```python
     print(json.dumps(result))
     ```  
   - Do not print anything else.  

8. **General Safety**  
   - Always validate columns exist before using them.  
   - Never assume all columns are numeric.  
   - Fail gracefully with a JSON error instead of an exception.  
   - Keep responses deterministic (set random_state if randomness is required).  

Example (scatterplot with regression line, shortened):

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import io, base64, json

df = pd.read_csv("data.csv")

# Safe column lookup
cols = {c.lower(): c for c in df.columns}
xcol = next((cols[c] for c in cols if "rank" in c), None)
ycol = next((cols[c] for c in cols if "peak" in c), None)

if not xcol or not ycol:
    result = {"error": "Missing required columns", "columns": df.columns.tolist()}
    print(json.dumps(result))
    exit()

x = df[xcol].dropna().values
y = df[ycol].dropna().values

m, b = np.polyfit(x, y, 1)
plt.scatter(x, y)
plt.plot(x, m*x + b, color="red", linestyle="dotted")

buf = io.BytesIO()
plt.savefig(buf, format="png", bbox_inches="tight")
plt.close()
plot_b64 = base64.b64encode(buf.getvalue()).decode()
plot_uri = f"data:image/png;base64,{plot_b64}"

result = [len(df), np.corrcoef(x, y)[0,1], plot_uri]
print(json.dumps(result))
"""

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
            model="gpt-4o", # This is a powerful and widely supported model
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