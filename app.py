from flask import Flask, request, jsonify, render_template
import joblib
import pandas as pd
from sklearn.preprocessing import LabelEncoder

app = Flask(__name__)

# Charger le modèle
model = joblib.load('trained_model.pkl')

# Ajouter toutes les catégories possibles pour la colonne 'Matière'
label_encoder = LabelEncoder()
label_encoder.fit(['PLASTIQUE', 'METAL', 'AUTRE', 'ALU'])

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    data = request.json
    qte_annuelle = data['qte_annuelle']
    longueur = data['longueur']
    largeur = data['largeur']
    epaisseur = data['epaisseur']
    hauteur = data['hauteur']
    matiere = data['matiere']
    masse = data['masse']

    if matiere not in label_encoder.classes_:
        return jsonify({'error': 'Matière non reconnue'}), 400

    matiere_encoded = label_encoder.transform([matiere])[0]

    new_reference = pd.DataFrame({
        'Qté annuelle': [qte_annuelle],
        'Longueur': [longueur],
        'Largeur': [largeur],
        'Epaisseur': [epaisseur],
        'Hauteur': [hauteur],
        'Matière': [matiere_encoded],
        'Masse (kg)': [masse]
    })

    predicted_price = model.predict(new_reference)
    return jsonify({'predicted_price': f"{predicted_price[0]:.2f} €"})

if __name__ == '__main__':
    app.run(debug=True)
