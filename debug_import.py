try:
    from sentence_transformers import SentenceTransformer
    print("Success")
except Exception as e:
    print(f"Error: {e}")
    import traceback
    traceback.print_exc()
