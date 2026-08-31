import os
from dotenv import load_dotenv
from huggingface_hub import get_token, login

# .env dosyasindaki degiskenleri sisteme yukler
load_dotenv()


def hugging_face_auth(token: str = None) -> bool:
    """
    Authenticates with Hugging Face Hub using an explicit token,
    the .env variable (HF_TOKEN), or existing cached local credentials.
    Returns True if authentication succeeded, False otherwise.
    """
    hf_token = (
        token
        or os.getenv("HF_TOKEN")
        or os.getenv("HUGGINGFACE_TOKEN")
        or os.getenv("HUGGING_FACE_TOKEN")
    )

    print("Attempting Hugging Face login...")
    try:
        if hf_token:
            login(token=hf_token)
            print("Login successful (via token)!")
            return True
        elif get_token() is not None:
            print("Login successful (using cached credentials)!")
            return True
        else:
            print("Notice: No HF_TOKEN provided and no cached credentials found. Public models will still work.")
            return False
    except Exception as e:
        print(f"Warning: Hugging Face authentication failed: {e}")
        return False


if __name__ == "__main__":
    hugging_face_auth()
