# benchmark/swebench/api_server.py
from fastapi import FastAPI, BackgroundTasks, HTTPException
from pydantic import BaseModel
from typing import List, Dict, Any
import uuid

app = FastAPI(title="SWE-bench Evaluation API")

class Submission(BaseModel):
    run_id: str
    model_name: str
    sparse_attention_config: Dict[str, Any]
    instance_id: str
    repo: str
    base_commit: str
    patch: str
    metadata: Dict[str, Any]

class BatchSubmission(BaseModel):
    benchmark: str
    num_submissions: int
    submissions: List[Submission]

@app.post("/submit")
async def receive_submissions(batch: BatchSubmission, background_tasks: BackgroundTasks):
    # Log the receipt
    print(f"Received {batch.num_submissions} submissions for {batch.benchmark}")
    
    # Logic to trigger SWE-bench docker evaluation goes here
    # background_tasks.add_task(run_swebench_eval, batch.submissions)
    
    return {"status": "accepted", "job_id": str(uuid.uuid4())}