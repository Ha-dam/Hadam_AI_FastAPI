from fastapi import FastAPI, HTTPException, Depends, Form
from pydantic import BaseModel
from sqlalchemy.ext.automap import automap_base
from sqlalchemy import create_engine, Column, Integer, String, Text, Boolean, DateTime, ForeignKey, Enum
from sqlalchemy.orm import sessionmaker,Session, relationship
from sqlalchemy.ext.declarative import declarative_base
from datetime import datetime
import requests
import base64
from dotenv import load_dotenv
import openai
import os

# 환경 변수 로드
load_dotenv()
DB_USER = os.getenv("DB_USER")
DB_PASS = os.getenv("DB_PASS")
DB_HOST = os.getenv("DB_HOST")
DB_NAME = os.getenv("DB_NAME")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
STABLE_DIFFUSION_API_KEY = os.getenv("STABLE_DIFFUSION_API_KEY")
API_HOST = os.getenv('API_HOST', 'https://api.stability.ai')


# 데이터베이스 연결 설정
DATABASE_URL = f"mysql+pymysql://{DB_USER}:{DB_PASS}@{DB_HOST}/{DB_NAME}"
engine = create_engine(DATABASE_URL)

# Automap Base 생성 및 기존 데이터베이스 테이블과의 클래스 매핑 준비
Base = automap_base()
Base.prepare(engine, reflect=True)

# 매핑된 클래스 참조 생성
Member = Base.classes.member
Diary = Base.classes.diary
DiaryInfo = Base.classes.diary_info

# 세션 생성을 위한 sessionmaker 설정
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)

# FastAPI 앱 인스턴스 생성
app = FastAPI()

# OpenAI GPT-3 설정
openai.api_key = OPENAI_API_KEY

# 데이터베이스 세션 의존성 주입을 위한 함수 추가
def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()

# Pydantic 모델 정의
class DiaryRequest(BaseModel):
    
    member_id: int
    date: str
    location: str
    people: str
    happy: int
    comfortable: int
    sad: int
    angry: int
    experience: str


# 그림 생성을 위한 요약 생성 함수
def generate_summary(date, location, people, experience):
    summary_instruct = [
        {
            "role": "system",
            "content": "Please generate a sentence using the given parameters that can be used as a title for a diary entry. "
                       "In 50 letters or less than 50 letters "
                       f"Date: {date}, Location: {location}, People: {people}, Experience: {experience}"
        }
    ]

    response = openai.ChatCompletion.create(
        model="ft:gpt-3.5-turbo-1106:personal::8RpLwK9R",
        messages=summary_instruct,
        max_tokens=100,
        temperature=0.5,
        top_p=1
    )
    summary = response.choices[0].message.content.strip()
    return summary


# 일기 생성 API 엔드포인트
@app.post("/diary/create")
async def create_diary(request: DiaryRequest, db: Session = Depends(get_db)):
    messages = [
        {
            "role": "system",
            "content": "You are a program that receives keywords related to time, location, people, and experience to generate diaries. "
                       "The main information provided will include emotional state scores (comfortable, happy, angry, sad), date, location, and people as basic information, "
                       "and the user will input text about their experiences that day.\n\nBased on the information provided, please write a diary. "
                       "The diary will be written in Korean."
                       "make the content longer and more natural"
                       "Don't talk about emotion's score"
        },
        {
            "role": "user",
            "content": (f"Date: {request.date}, Location: {request.location}, People: {request.people}, "
                        f"Happy: {request.happy}, Comfortable: {request.comfortable}, Sad: {request.sad}, "
                        f"Angry: {request.angry}, Experience: {request.experience}")
        }
    ]
    
    try:
        response = openai.ChatCompletion.create(
            model="ft:gpt-3.5-turbo-1106:personal::8RgY8JYa",
            messages = messages,
            temperature=1,
            max_tokens=1000,
            top_p=1,
            frequency_penalty=0.7,
            presence_penalty=0
        )
        content = response.choices[0].message.content.strip()
        
        # 요약 생성
        summary_content = generate_summary(request.date, request.location, request.people, request.experience)

        # Stable Diffusion API 호출 및 이미지 생성
        response = requests.post(
            f"{API_HOST}/v1/generation/stable-diffusion-v1-6/text-to-image",
            headers={
                "Content-Type": "application/json",
                "Accept": "application/json",
                "Authorization": f"Bearer {STABLE_DIFFUSION_API_KEY}"
            },
            json={
                "text_prompts":
                [{
                    "text": "With No people," + summary_content, 
                    "weight": 1
                },
                {
                    "text": "blurry, bad quality, disfigured, kitsch, ugly, oversaturated, greain, low-res, deformed, violence, gore, blood",
                    "weight": -1
                }],
                "cfg_scale": 7,
                "height": 512,
                "width": 512,
                "samples": 1,
                "steps": 40,
            },
        )

        if response.status_code != 200:
            raise Exception("Non-200 response from Stable Diffusion API: " + str(response.text))

        # 이미지 데이터 처리: base64형태의 text로 저장, 안드로이드에서 디코딩 필요
        data = response.json()
        image_data = base64.b64decode(data["artifacts"][0]["base64"])
        base64_image = base64.b64encode(image_data).decode()



        # 현재 시간 가져오기
        current_time = datetime.now()

        # Diary 인스턴스 생성 및 저장
        new_diary = Diary()
        new_diary.content = content # GPT-3 응답으로부터 생성된 내용
        new_diary.member_id = request.member_id
        new_diary.img = base64_image
        new_diary.is_liked =False
        new_diary.created_at = current_time # 생성 시간 설정
        new_diary.modified_at = current_time # 수정 시간 설정

        db.add(new_diary)
        db.commit()
        db.refresh(new_diary)

        # DiaryInfo 인스턴스 생성 및 저장
        new_diary_info = DiaryInfo()
        new_diary_info.diary_id = new_diary.id
        new_diary_info.created_at = new_diary.created_at
        new_diary_info.modified_at = new_diary.modified_at
        new_diary_info.angry= request.angry
        new_diary_info.comfortable=request.comfortable
        new_diary_info.date=request.date
        new_diary_info.experience=request.experience
        new_diary_info.happy=request.happy
        new_diary_info.person=request.people
        new_diary_info.place=request.location
        new_diary_info.sad=request.sad
  
        db.add(new_diary_info)
        db.commit()
        db.refresh(new_diary_info)
        
        return {"content": content, "diary_id": new_diary.id, "image_data": base64_image}
    except Exception as e:
        db.rollback()
        raise HTTPException(status_code=500, detail=str(e))



#일기(텍스트) 재생성 API 앤드포인트
@app.put("/diary/recreate/{diary_id}")
async def recreate_diary(diary_id: int, db: Session = Depends(get_db)):
    try:
        # 기존 일기와 관련된 DiaryInfo 조회
        diary_info = db.query(DiaryInfo).filter(DiaryInfo.diary_id == diary_id).first()
        if not diary_info:
            raise HTTPException(status_code=404, detail=f"DiaryInfo for diary id {diary_id} not found")

        #  diary_info의 값을 이용해 GPT-3 API 호출
        gpt_messages = [
            {
                "role": "system",
                "content": "You are a program that receives keywords related to time, location, people, and experience to generate diaries. "
                       "The main information provided will include emotional state scores (comfortable, happy, angry, sad), date, location, and people as basic information, "
                       "and the user will input text about their experiences that day.\n\nBased on the information provided, please write a diary. "
                       "The diary will be written in Korean."
                       "make the content longer and more natural"
                       "Don't talk about emotion's score"
        
            },
            {
                "role": "user",
                "content": (f"Date: {diary_info.date}, Location: {diary_info.place}, People: {diary_info.person}, "
                            f"Happy: {diary_info.happy}, Comfortable: {diary_info.comfortable}, Sad: {diary_info.sad}, "
                            f"Angry: {diary_info.angry}, Experience: {diary_info.experience}")
            }
        ]

        gpt_response = openai.ChatCompletion.create(
            model="ft:gpt-3.5-turbo-1106:personal::8RgY8JYa",
            messages=gpt_messages,
            temperature=1,
            max_tokens=1000,
            top_p=1,
            frequency_penalty=0.7,
            presence_penalty=0
        )
        new_content = gpt_response.choices[0].message.content.strip()

        # Diary 내용 업데이트
        diary = db.query(Diary).filter(Diary.id == diary_id).first()
        if not diary:
            raise HTTPException(status_code=404, detail=f"Diary with id {diary_id} not found")

        diary.content = new_content
        diary.modified_at = datetime.now()
        db.commit()

        return {"message": "Diary content updated successfully", "diary_id": diary_id}
    except Exception as e:
        db.rollback()
        raise HTTPException(status_code=500, detail=str(e))




#그림 재생성 API 앤드포인트
@app.put("/diary/regenerate-image/{diary_id}")
async def regenerate_diary_image(diary_id: int, db: Session = Depends(get_db)):
    try:
        # 기존 일기 조회
        diary = db.query(Diary).filter(Diary.id == diary_id).first()
        if not diary:
            raise HTTPException(status_code=404, detail=f"Diary with id {diary_id} not found")
        
        # 관련된 DiaryInfo 조회
        diary_info = db.query(DiaryInfo).filter(DiaryInfo.diary_id == diary_id).first()
        if not diary_info:
            raise HTTPException(status_code=404, detail=f"DiaryInfo for diary id {diary_id} not found")
        
        # 요약 생성
        summary_content = generate_summary(diary_info.date, diary_info.place, diary_info.person, diary_info.experience)


        # Stable Diffusion API 호출 및 이미지 생성
        response = requests.post(
            f"{API_HOST}/v1/generation/stable-diffusion-v1-6/text-to-image",
            headers={
                "Content-Type": "application/json",
                "Accept": "application/json",
                "Authorization": f"Bearer {STABLE_DIFFUSION_API_KEY}"
            },
            json={
                "text_prompts":
                [{
                    "text": "With No people," + summary_content, 
                    "weight": 1
                },
                {
                    "text": "blurry, bad quality, disfigured, kitsch, ugly, oversaturated, greain, low-res, deformed, violence, gore, blood",
                    "weight": -1
                }],
                "cfg_scale": 7,
                "height": 512,
                "width": 512,
                "samples": 1,
                "steps": 40,
            },
        )

        if response.status_code != 200:
            raise Exception("Non-200 response from Stable Diffusion API: " + str(response.text))

        # 이미지 데이터 처리: base64형태의 text로 저장, 안드로이드에서 디코딩 필요
        data = response.json()
        image_data = base64.b64decode(data["artifacts"][0]["base64"])
        base64_image = base64.b64encode(image_data).decode()

        # Diary 이미지 업데이트
        diary.img = base64_image
        diary.modified_at = datetime.now()
        db.commit()

        return {"message": "Diary image updated successfully", "diary_id": diary_id}
    except Exception as e:
        db.rollback()
        raise HTTPException(status_code=500, detail=str(e))
