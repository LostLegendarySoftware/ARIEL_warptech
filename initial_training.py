import json
from core.encryption import CoreEncryption
from core.ariel_training import train_ariel  # This would be decrypted and imported

def initial_training(user_data):
    # Load and decrypt core files
    encryptor = CoreEncryption()
    ariel_core = encryptor.decrypt_file('ariel_algorithm.py.enc')
    
    # Load base intelligence
    with open('data/base_intelligence.json', 'r') as file:
        base_intelligence = json.load(file)
    
    # Combine base intelligence with user data
    training_data = {**base_intelligence, **user_data}
    
    # Run initial training
    ariel_instance = train_ariel(training_data)
    
    return ariel_instance

# This function would be called during first-time setup