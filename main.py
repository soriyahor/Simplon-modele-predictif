import pickle
from fastapi import FastAPI, HTTPException
import uvicorn

app = FastAPI()

# Fonction pour charger le modèle
def load_model(file_path):
    try:
        with open(file_path, 'rb') as file:
            model = pickle.load(file)
            return model
    except FileNotFoundError:
        print('fichier modèle introuvable')
        return None

# Fonction pour effectuer la prédiction
def predict(model, longitude, latitude):
    if model is None:
        raise HTTPException(status_code=503, detail="Modèle non chargé")
    input_data = [[longitude, latitude]]
    return model.predict(input_data)[0]

def predict_two(model, n_pieces, surface_habitable, prix_m2, code_postal, longitude, latitude):
    if model is None:
        raise HTTPException(status_code=503, detail="Modèle non chargé")
    input_data = [[n_pieces, surface_habitable, prix_m2, code_postal, longitude, latitude]]
    return model.predict(input_data)[0]

# Route pour effectuer la prédiction
@app.post('/prix_m2', description="Permet de donner une prédiction du prix au mètre carré en fonction de longitude et la latitude")
def prix_m2(longitude: float, latitude: float):
    # Chargement du modèle à chaque appel
    loaded_model = load_model('optimal_rfr_model.pkl')
    
    # Vérification du chargement du modèle
    if loaded_model is None:
        raise HTTPException(status_code=503, detail="Modèle non chargé")

    # Prédiction
    prediction = predict(loaded_model, longitude, latitude)
    return {'predicted_prix_m2': prediction}

@app.post('/prix_m2_two', description="Permet de donner une prédiction du prix au mètre carré en fonction du nombre de pièce, de la surface habitable, du code postal, de la longitude et la latitude")
def prix_m2_two(n_pieces: int, surface_habitable: float, prix_m2: float, code_postal: int, longitude: float, latitude: float):
    # Chargement du modèle à chaque appel
    loaded_model = load_model('optimal_linear_regression_model.pkl')
    
    # Vérification du chargement du modèle
    if loaded_model is None:
        raise HTTPException(status_code=503, detail="Modèle non chargé")

    # Prédiction
    prediction = predict_two(loaded_model, n_pieces, surface_habitable, prix_m2, code_postal, longitude, latitude)
    
    return {'predicted_prix_m2': prediction}

uvicorn.run(app)


