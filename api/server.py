from fastapi import FastAPI, HTTPException
from typing import Dict, Any
from environment.env import MedicalTriageEnv

app = FastAPI(title="Medical Triage Environment API")
env = MedicalTriageEnv()

@app.get("/")
def root():
    return {"message": "API is running"}

@app.get("/health")
def health():
    return {"status": "ok"}

@app.post("/reset")
def reset():
    try:
        obs = env.reset()
        return obs.dict()
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/step")
def step(action: dict):
    try:
        obs, reward, done, info = env.step(action)
        return {
            "observation": obs.dict(),
            "reward": reward,
            "done": done,
            "info": info
        }
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

@app.get("/state")
def state():
    try:
        obs = env.state()
        return obs.dict()
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/tasks")
def tasks():
    return {"tasks": list(env.tasks.keys())}

@app.post("/reset/{task_name}")
def reset_task(task_name: str):
    if task_name not in env.tasks:
        raise HTTPException(status_code=404, detail="Task not found")
    try:
        obs = env.reset(task_name=task_name)
        return obs.dict()
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
