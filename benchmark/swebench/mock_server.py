from fastapi import FastAPI, Request
import uvicorn

app = FastAPI()

@app.post("/submit")
async def receive_submission(request: Request):
    data = await request.json()
    
    # Verification prints
    print("\n" + "="*50)
    print("📥 RECEIVED SUBMISSION FROM EXECUTOR")
    print(f"Model: {data.get('model_name', 'Unknown')}")
    print(f"Patches received: {len(data.get('submissions', []))}")
    
    # Print the first patch preview
    if data.get('submissions'):
        first = data['submissions'][0]
        print(f"First Instance ID: {first.get('instance_id')}")
        print(f"Patch Preview: {first.get('patch')[:100]}...")
    print("="*50 + "\n")
    
    return {"status": "success", "message": "Data received by mock server"}

if __name__ == "__main__":
    uvicorn.run(app, host="127.0.0.1", port=8000)