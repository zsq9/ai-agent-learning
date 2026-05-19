from fastapi import FastAPI
from pydantic import BaseModel



app = FastAPI(title="AI Agent 学习第一天服务")


@app.get("/")
def home():
    return {"message" : "AI Agent 学习启动！"}

@app.get("/user/{name}")
def get_user(name:str):
    return{
        "username":name,
        "status":"正在学习 AI Agent",
        "msg":"学习中..."
    }

class UserRequest(BaseModel):
    question:str
    user_id:str
@app.post("/agent/chat")
def agent_chat(req:UserRequest):
    print(type(req))
    print(req)
    print(req.model_dump())
    return{
        "user_id":req.user_id,
        "你的问题":req.question,
        "AI回复":"正在学习中，稍后为你解答！"
    }
