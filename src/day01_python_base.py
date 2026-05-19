import json
from dotenv import load_dotenv
import os


user_info={
    "name" : "zhangsan",
    "age" : 25,
    "skills" : ["python" , "FastAPI"]
}
print("用户信息",user_info)
print("年龄",user_info["age"])
print("技能",user_info["skills"])
print(type(user_info),type(user_info["age"]),type(user_info["skills"]))
#-------------------------------------------------------------
def say_hello(name:str):
    return f"hello,{name}"
print(say_hello("Z17"))
#----------------------------------------
json_str = json.dumps(user_info,ensure_ascii=False,indent=2)
print("JSON格式数据:\n",json_str)
data=json.loads(json_str)
print("转回字典:",data)
#-------------------------------------------
try:
    print(10/0)
except ZeroDivisionError as e:
    print("程序出错原因：",e)
# ---------------------------------
load_dotenv()

