import os

def simulate(file_name, process_like_image=1):
    input_folder = os.path.join(os.path.dirname(os.path.realpath(__file__)), "inputs")
    ouput_folder = os.path.join(os.path.dirname(os.path.realpath(__file__)), "outputs")
    encrpyted_folder = os.path.join(os.path.dirname(os.path.realpath(__file__)), "encrypted")
    keys_folder = os.path.join(os.path.dirname(os.path.realpath(__file__)), "keys")

    # file and encryption key
    file = os.path.join(input_folder, file_name)
    correct_key = os.path.join(keys_folder, "correct.json")
    incorrect_key = os.path.join(keys_folder, "incorrect.json")

    # Encrypt with the given key
    print("Encrypting...")
    os.system(f"python encrypt_file.py {file} {correct_key} {process_like_image} {encrpyted_folder}")

    # Decrypt with same key
    print("Decrypting with the same key")
    os.system(f"python decrypt_file.py {correct_key} {process_like_image} {encrpyted_folder} {ouput_folder}")

    # Decrypt with different key
    print("Decrypting with a slightly different key")
    os.system(f"python decrypt_file.py {incorrect_key} {process_like_image} {encrpyted_folder} {ouput_folder}")

if __name__ == "__main__":

    # simulate for sample files
    simulate(file_name="cat.png", process_like_image=1)
    simulate(file_name="lena.tif", process_like_image=1)
    simulate(file_name="sample_audio.mp3", process_like_image=0)