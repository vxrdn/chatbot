from fastapi import FastAPI, HTTPException, Depends, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Optional, Dict, Any
import openai
import os
from sqlalchemy import create_engine, Column, Integer, String, Float, Text, DateTime
from sqlalchemy.orm import sessionmaker, Session, declarative_base
import datetime
import nltk
from nltk.sentiment import SentimentIntensityAnalyzer
import json
# Replace openai import with Google's package
import google.generativeai as genai
import os
# Replace OpenAI setup with Gemini setup
# filepath: c:\Users\KIIT\Desktop\coding\160223\java\.vscode\daa lab\project\backend.py
# ...existing code...
# Set your Google API key
genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))
# ...existing code...

# Download NLTK resources
try:
    nltk.data.find('vader_lexicon')
except LookupError:
    nltk.download('vader_lexicon')

# Initialize FastAPI
app = FastAPI(title="AI Chatbot Optimization API")

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Restrict in production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Database setup
engine = create_engine(os.getenv("DATABASE_URL"))
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
Base = declarative_base()

# Database Models
class Prompt(Base):
    __tablename__ = "prompts"
    
    id = Column(Integer, primary_key=True, index=True)
    content = Column(Text, nullable=False)
    version = Column(Integer, default=1)
    performance_score = Column(Float, default=0.0)
    created_at = Column(DateTime, default=datetime.datetime.utcnow)
    
class Conversation(Base):
    __tablename__ = "conversations"
    
    id = Column(Integer, primary_key=True, index=True)
    user_id = Column(String, index=True)
    prompt_id = Column(Integer)
    user_message = Column(Text)
    bot_response = Column(Text)
    user_feedback = Column(Float, nullable=True)
    created_at = Column(DateTime, default=datetime.datetime.utcnow)

# Create the database tables
Base.metadata.create_all(bind=engine)

# Dependency
def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()

# Request and Response Models
class MessageRequest(BaseModel):
    user_id: str
    message: str
    prompt_id: Optional[int] = None

class MessageResponse(BaseModel):
    response: str
    prompt_id: int

class FeedbackRequest(BaseModel):
    conversation_id: int
    feedback_score: float
    feedback_text: Optional[str] = None

# Helper Functions
def get_optimized_prompt(db: Session, prompt_id: Optional[int] = None):
    """Get the best performing prompt or a specific prompt"""
    if prompt_id:
        return db.query(Prompt).filter(Prompt.id == prompt_id).first()
    return db.query(Prompt).order_by(Prompt.performance_score.desc()).first()

def generate_ai_response(message: str, prompt_content: str, db: Session = None, user_id: str = None):
    """Generate response from Gemini API with conversation history"""
    try:
        # Configure the model
        generation_config = {
            "temperature": 0.7,
            "top_p": 0.95,
            "top_k": 40,
            "max_output_tokens": 1024,
        }
        
        # Get a specific model (gemini-pro is suitable for text)
        model = genai.GenerativeModel(
            model_name="gemini-1.5-flash",
            generation_config=generation_config
        )
        
        # Add conversation history if available
        conversation_history = ""
        if db and user_id:
            conversation_history = get_user_conversation_history(db, user_id)
        
        # Create the prompt by combining system prompt, conversation history, and user message
        if conversation_history:
            combined_prompt = f"{prompt_content}\n\nPrevious conversation:\n{conversation_history}\n\nUser: {message}"
        else:
            combined_prompt = f"{prompt_content}\n\nUser: {message}"
        
        # Generate the response
        response = model.generate_content(combined_prompt)
        
        # Extract text from response
        return response.text
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"AI model error: {str(e)}")

def analyze_response_quality(response: str):
    """Basic analysis of response quality using NLTK"""
    analyzer = SentimentIntensityAnalyzer()
    sentiment = analyzer.polarity_scores(response)
    
    # Example criteria (expand as needed)
    word_count = len(response.split())
    quality_score = 0.5  # Default middle score
    
    # Adjust based on length (simple example)
    if 20 <= word_count <= 200:
        quality_score += 0.2
    
    # Adjust based on sentiment
    if sentiment['compound'] > 0:
        quality_score += 0.1
    
    return min(quality_score, 1.0)  # Cap at 1.0

def optimize_prompt(db: Session, prompt_id: int):
    """Use feedback data to optimize a prompt"""
    # Get current prompt
    prompt = db.query(Prompt).filter(Prompt.id == prompt_id).first()
    if not prompt:
        return
    
    # Get all conversation feedback for this prompt
    conversations = db.query(Conversation).filter(
        Conversation.prompt_id == prompt_id,
        Conversation.user_feedback.isnot(None)
    ).all()
    
    if not conversations:
        return
    
    # Calculate average feedback
    avg_feedback = sum(c.user_feedback for c in conversations) / len(conversations)
    
    # Update prompt performance score
    prompt.performance_score = avg_feedback
    db.commit()
    
    # If score is low, create an improved version (could use OpenAI to help optimize)
    if avg_feedback < 0.6 and len(conversations) > 5:
        # Create a new improved prompt version
        try:
            generation_config = {
                "temperature": 0.7,
                "max_output_tokens": 1024,
            }
            
            model = genai.GenerativeModel(
                model_name="gemini-pro",
                generation_config=generation_config
            )
            
            prompt_improvement_request = f"""You are a prompt optimization expert.
            
            This prompt needs improvement: '{prompt.content}'.
            Users gave it a score of {avg_feedback}/1.
            Please create an improved version."""
            
            improvement_response = model.generate_content(prompt_improvement_request)
            
            new_prompt = Prompt(
                content=improvement_response.text,
                version=prompt.version + 1,
                performance_score=0.0  # Will be updated based on future feedback
            )
            db.add(new_prompt)
            db.commit()
        except Exception as e:
            print(f"Failed to optimize prompt: {str(e)}")

def get_user_conversation_history(db: Session, user_id: str, limit: int = 5):
    """Get recent conversation history for a user"""
    previous_conversations = (
        db.query(Conversation)
        .filter(Conversation.user_id == user_id)
        .order_by(Conversation.created_at.desc())
        .limit(limit)
        .all()
    )
    
    # Format conversations as a string
    conversation_history = ""
    for conv in reversed(previous_conversations):  # Reverse to get chronological order
        conversation_history += f"User: {conv.user_message}\n"
        conversation_history += f"Assistant: {conv.bot_response}\n\n"
    
    return conversation_history

# API Endpoints
@app.post("/chat", response_model=MessageResponse)
def chat(message_request: MessageRequest, db: Session = Depends(get_db)):
    """Process a user message and return an AI response"""
    
    # Get the appropriate prompt
    prompt = get_optimized_prompt(db, message_request.prompt_id)
    if not prompt:
        # Create a default prompt if none exists
        prompt = Prompt(
            content="You are a helpful AI assistant that provides clear and concise responses.",
            version=1
        )
        db.add(prompt)
        db.commit()
        db.refresh(prompt)
    
    # Generate AI response with conversation history
    response = generate_ai_response(
        message_request.message, 
        prompt.content,
        db=db,
        user_id=message_request.user_id
    )
    
    # Save the conversation
    conversation = Conversation(
        user_id=message_request.user_id,
        prompt_id=prompt.id,
        user_message=message_request.message,
        bot_response=response
    )
    db.add(conversation)
    db.commit()
    db.refresh(conversation)
    
    return {"response": response, "prompt_id": prompt.id}

@app.post("/feedback")
def submit_feedback(feedback_request: FeedbackRequest, background_tasks: BackgroundTasks, db: Session = Depends(get_db)):
    """Submit feedback for a conversation and trigger prompt optimization"""
    
    # Get the conversation
    conversation = db.query(Conversation).filter(Conversation.id == feedback_request.conversation_id).first()
    if not conversation:
        raise HTTPException(status_code=404, detail="Conversation not found")
    
    # Update feedback
    conversation.user_feedback = feedback_request.feedback_score
    db.commit()
    
    # Trigger prompt optimization in background
    background_tasks.add_task(optimize_prompt, db, conversation.prompt_id)
    
    return {"status": "Feedback submitted successfully"}

@app.get("/prompts", response_model=List[Dict[str, Any]])
def get_prompts(db: Session = Depends(get_db)):
    """Get all prompts with their performance metrics"""
    prompts = db.query(Prompt).all()
    return [{"id": p.id, "content": p.content, "version": p.version, "score": p.performance_score} for p in prompts]

# Run the application
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)