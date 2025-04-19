from cryptography.fernet import Fernet
import os

def decrypt_file(key, encrypted_file_path):
    with open(key, 'rb') as key_file:
        key = key_file.read()

    f = Fernet(key)

    with open(encrypted_file_path, 'rb') as encrypted_file:
        encrypted_data = encrypted_file.read()

    decrypted_data = f.decrypt(encrypted_data)

    decrypted_file_path = encrypted_file_path[:-4]  # Remove .enc extension
    with open(decrypted_file_path, 'wb') as decrypted_file:
        decrypted_file.write(decrypted_data)

    print(f"Decrypted: {encrypted_file_path} -> {decrypted_file_path}")

# Path to your key file
key_file = 'core_key.key'

# List of encrypted files
encrypted_files = [
    'ariel_training.py.enc',
    'train_ariel.py.enc',
    'ariel_algorithm.py.enc',
    # Add any other encrypted files here
]

# Decrypt all files
for file in encrypted_files:
    decrypt_file(key_file, file)