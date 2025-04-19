from cryptography.fernet import Fernet
import os

def encrypt_file(key, file_path):
    with open(key, 'rb') as key_file:
        key = key_file.read()

    f = Fernet(key)

    with open(file_path, 'rb') as file:
        file_data = file.read()

    encrypted_data = f.encrypt(file_data)

    encrypted_file_path = file_path + '.enc'
    with open(encrypted_file_path, 'wb') as encrypted_file:
        encrypted_file.write(encrypted_data)

    print(f"Encrypted: {file_path} -> {encrypted_file_path}")

# Path to your key file
key_file = 'core_key.key'

# List of files to encrypt
files_to_encrypt = [
    'ariel_training.py',
    'train_ariel.py',
    'ariel_algorithm.py',
    # Add any other files you want to encrypt here
]

# Encrypt all files
for file in files_to_encrypt:
    if os.path.exists(file):
        encrypt_file(key_file, file)
    else:
        print(f"Warning: {file} not found. Skipping.")

print("Encryption complete. Remember to securely store your core_key.key file!")