import sys
import os

print(f"🐍 Python Executable: {sys.executable}")
print(f"📂 Working Directory: {os.getcwd()}")

try:
    import redis
    print(f"✅ Redis imported successfully!")
    print(f"📦 Redis Version: {redis.__version__}")
    print(f"📍 Redis Location: {redis.__file__}")
except ImportError as e:
    print(f"❌ Failed to import redis: {e}")
    # 打印搜索路径帮助排查
    print("🔍 Sys Path:")
    for p in sys.path:
        print(f"  - {p}")
except Exception as e:
    print(f"❌ Other error: {e}")
