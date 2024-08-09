from huggingface_hub import login
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
import sqlite3
from flask import Flask, request, jsonify
import requests
from sqlalchemy import create_engine
from sqlalchemy.pool import StaticPool
import sqlparse
from langchain_community.utilities.sql_database import SQLDatabase


# Access the Hugging Face token from Kaggle secrets
huggingface_token = "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"

# Log in to Hugging Face
login(huggingface_token)

model_name = "defog/sqlcoder-7b-2"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(
        model_name,
        trust_remote_code=True,
        torch_dtype=torch.float16,
        device_map="auto",
        use_cache=True,
    )


def get_engine_for_chinook_db():
    """Pull sql file, populate in-memory database, and create engine."""
    url = "https://raw.githubusercontent.com/lerocha/chinook-database/master/ChinookDatabase/DataSources/Chinook_Sqlite.sql"
    response = requests.get(url)
    sql_script = response.text

    connection = sqlite3.connect(":memory:", check_same_thread=False)
    connection.executescript(sql_script)
    return create_engine(
        "sqlite://",
        creator=lambda: connection,
        poolclass=StaticPool,
        connect_args={"check_same_thread": False},
    )


prompt = """### Task
            Generate a SQL query to answer [QUESTION]{question}[/QUESTION]

            ### Instructions
            - If you cannot answer the question with the available database schema, return 'I do not know'
            - Remember that revenue is price multiplied by quantity
            - Remember that cost is supply_price multiplied by quantity

            ### Database Schema
            This query will run on a database whose schema is represented in this string:
            {schema}

            ### Answer
            Given the database schema, here is the SQL query that answers [QUESTION]{question}[/QUESTION]
            [SQL]
        """


def generate_query(question , schema):
    updated_prompt = prompt.format(question=question , schema = schema)
    inputs = tokenizer(updated_prompt, return_tensors="pt").to("cuda")
    generated_ids = model.generate(
        **inputs,
        num_return_sequences=1,
        eos_token_id=tokenizer.eos_token_id,
        pad_token_id=tokenizer.eos_token_id,
        max_new_tokens=400,
        do_sample=False,
        num_beams=1,
    )
    outputs = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)

    torch.cuda.empty_cache()
    torch.cuda.synchronize()
    # empty cache so that you do generate more results w/o memory crashing
    # particularly important on Colab – memory management is much more straightforward
    # when running on an inference service
    return sqlparse.format(outputs[0].split("[SQL]")[-1], reindent=True)

engine = get_engine_for_chinook_db()
db = SQLDatabase(engine)
schema = db.table_info

app = Flask(__name__)

@app.route("/")
def home():
    return "Welcome to SQL Query Generator!"


@app.route('/generate-sql', methods=['POST'])
def generate_sql():
    """API endpoint to generate SQL query."""
    data = request.json
    question = data.get('question')
    

    if not question or not schema:
        return jsonify({'error': 'Question and schema are required'}), 400

    try:
        sql_query = generate_query(question, schema)
        return jsonify({'sql_query': sql_query})
    except Exception as e:
        return jsonify({'error': str(e)}), 500


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)