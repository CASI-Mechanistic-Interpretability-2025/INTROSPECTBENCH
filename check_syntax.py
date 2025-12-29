import ast
import traceback

file_path = r"c:\Users\bucke\Documents\INTROSPECTBENCH\run_benchmark.py"

try:
    with open(file_path, "r", encoding="utf-8") as f:
        content = f.read()
    ast.parse(content)
    print("Syntax is OK")
except SyntaxError as e:
    print(f"SyntaxError: {e}")
    print(f"Line: {e.lineno}")
    print(f"Offset: {e.offset}")
    print(f"Text: {e.text}")
except Exception as e:
    traceback.print_exc()
