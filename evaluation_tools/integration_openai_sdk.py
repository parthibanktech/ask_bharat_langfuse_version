# Put this at the very top BEFORE importing langfuse.openai
import os

# If you're running in Streamlit, import streamlit and prefer st.secrets
try:
    import streamlit as st
    IS_STREAMLIT = True
except Exception:
    IS_STREAMLIT = False

# Optionally load .env for local dev (install python-dotenv if needed)
try:
    from dotenv import load_dotenv
    load_dotenv()  # loads .env into os.environ if .env exists
except Exception:
    pass

def configure_langfuse_from_streamlit():
    """
    Preference order:
      1. Streamlit secrets (st.secrets)
      2. Environment variables (os.environ)
      3. Interactive Streamlit input (only if nothing else found and running in Streamlit)
    This function will set os.environ variables expected by the langfuse SDK.
    """
    public = None
    secret = None
    base = None

    # 1) Try Streamlit secrets (Streamlit Cloud)
    if IS_STREAMLIT:
        # recommended structure in Streamlit secrets: {"LANGFUSE_PUBLIC_KEY": "...", "LANGFUSE_SECRET_KEY": "...", "LANGFUSE_BASE_URL": "..."}
        public = st.secrets.get("LANGFUSE_PUBLIC_KEY") or st.secrets.get("langfuse", {}).get("public_key")
        secret = st.secrets.get("LANGFUSE_SECRET_KEY") or st.secrets.get("langfuse", {}).get("secret_key")
        base = st.secrets.get("LANGFUSE_BASE_URL") or st.secrets.get("langfuse", {}).get("base_url")

    # 2) Fallback to environment variables
    if not public:
        public = os.getenv("LANGFUSE_PUBLIC_KEY")
    if not secret:
        secret = os.getenv("LANGFUSE_SECRET_KEY")
    if not base:
        base = os.getenv("LANGFUSE_BASE_URL")

    # 3) Optional: If running in Streamlit and still missing keys, allow interactive entry (temporary)
    if IS_STREAMLIT and (not public or not secret):
        st.warning("Langfuse keys not found in Streamlit secrets or environment. Enter them for this session only.")
        public = st.text_input("LANGFUSE_PUBLIC_KEY", value=public or "", type="password") or public
        secret = st.text_input("LANGFUSE_SECRET_KEY", value=secret or "", type="password") or secret
        base = st.text_input("LANGFUSE_BASE_URL", value=base or "https://cloud.langfuse.com")

    # Set into os.environ (langfuse integration reads env vars)
    if public:
        os.environ["LANGFUSE_PUBLIC_KEY"] = public
    if secret:
        os.environ["LANGFUSE_SECRET_KEY"] = secret
    if base:
        os.environ["LANGFUSE_BASE_URL"] = base

    # Optional: small check / notification (do not print secrets)
    if IS_STREAMLIT:
        if public and secret:
            st.success("Langfuse keys configured for this session.")
        else:
            st.error("Langfuse keys missing — please add them to Streamlit secrets or environment variables.")

# Call the configuration before importing Langfuse/OpenAI integration
configure_langfuse_from_streamlit()

# NOW import the Langfuse-wrapped OpenAI client
from langfuse.openai import openai  # this will pick up LANGFUSE_* from os.environ

# Continue with your code (examples below)
completion = openai.chat.completions.create(
    name="test-chat",
    model="gpt-4o",
    messages=[
        {"role": "system", "content": "You are a very accurate calculator. You output only the result of the calculation."},
        {"role": "user", "content": "1 + 1 = "}
    ],
    temperature=0,
    metadata={"someMetadataKey": "someValue"},
)

print(completion.choices[0].message["content"])
