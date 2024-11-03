# main.py
from fastapi import FastAPI
from pydantic import BaseModel
from typing import Optional
import requests
from difflib import get_close_matches
from data import training_data
from fastapi.middleware.cors import CORSMiddleware

app = FastAPI()

# Configuración CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class RecommendationRequest(BaseModel):
    user_input: str

class RecommendationResponse(BaseModel):
    response: str

def find_closest_match(user_input: str) -> Optional[str]:
    """Encuentra la entrada de training_data más parecida al user_input."""
    inputs = [item["user_input"] for item in training_data]
    closest_matches = get_close_matches(user_input, inputs, n=1, cutoff=0.6)
    if closest_matches:
        matched_input = closest_matches[0]
        for item in training_data:
            if item["user_input"] == matched_input:
                return item["response"]
    return None

def get_huggingface_model_response(user_input: str) -> str:
    """Función para obtener respuesta del modelo de Hugging Face."""
    try:
        response = requests.post(
            "https://api-inference.huggingface.co/models/meta-llama/Llama-3.2-1B",
            headers={
                "Content-Type": "application/json",
                "Authorization": "Bearer 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"  # Reemplaza con tu token
            },
            json={"inputs": user_input}
        )
        response.raise_for_status()  # Lanza un error si la respuesta no es 200

        # Verifica si la respuesta es una lista y obtiene el primer elemento
        result = response.json()
        if isinstance(result, list) and len(result) > 0:
            return result[0].get("generated_text", "No tengo una respuesta en este momento.")
        elif isinstance(result, dict):
            return result.get("generated_text", "No tengo una respuesta en este momento.")
        else:
            return "No tengo una respuesta en este momento."
    except Exception as e:
        return f"Error al conectar con el modelo de Hugging Face: {str(e)}"


@app.post("/recommendations", response_model=RecommendationResponse)
async def get_recommendations(request: RecommendationRequest):
    # Intenta encontrar una coincidencia en los datos de entrenamiento
    closest_response = find_closest_match(request.user_input)
    if closest_response:
        return RecommendationResponse(response=closest_response)
    
    # Si no se encuentra una coincidencia cercana, llama al modelo de Hugging Face
    fallback_response = get_huggingface_model_response(request.user_input)
    return RecommendationResponse(response=fallback_response)

@app.get("/")
async def root():
    return {"message": "Bienvenido a la API de recomendaciones de productos"}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=8000)