#!/usr/bin/env python3
"""
Setup script to create .env file for the Regulator project.
"""

import os
from pathlib import Path


def create_env_file():
    """Create a .env file with default configuration."""

    project_root = Path(__file__).parent.parent
    env_file = project_root / ".env"

    if env_file.exists():
        print(f"✅ .env file already exists at {env_file}")
        return

    env_content = """# Regulator Environment Configuration
# Fill in your actual values below

# OpenAI API Configuration
OPENAI_API_KEY=your_openai_api_key_here
OPENAI_MODEL=gpt-4o-mini

# LLM Configuration
LLM_TEMPERATURE=0.7
LLM_MAX_TOKENS=100
LLM_REQUEST_DELAY=0.5

# Collusion Detection
COLLUSION_CONFIDENCE_THRESHOLD=0.7
COLLUSION_DETECTION_ENABLED=true

# Message Generation
MESSAGE_FREQUENCY=0.3
CONVERSATION_MEMORY_SIZE=10

# Rate Limiting
API_RATE_LIMIT=10  # requests per minute
API_TIMEOUT=30     # seconds

# Logging
LOG_LEVEL=INFO
LOG_FILE=logs/regulator.log

# Dashboard
DASHBOARD_PORT=8501
DASHBOARD_HOST=localhost
"""

    try:
        with open(env_file, "w") as f:
            f.write(env_content)

        print(f"✅ Created .env file at {env_file}")
        print("📝 Please edit the .env file and add your OpenAI API key:")
        print("   OPENAI_API_KEY=your_actual_api_key_here")
        print()
        print("🔧 To get an OpenAI API key:")
        print("   1. Go to https://platform.openai.com/api-keys")
        print("   2. Create a new API key")
        print("   3. Copy it to your .env file")
        print()
        print("🚀 After setting up your API key, you can run:")
        print("   python scripts/real_llm_messages.py")

    except Exception as e:
        print(f"❌ Error creating .env file: {e}")


def check_env_setup():
    """Check if environment is properly configured."""

    from dotenv import load_dotenv

    load_dotenv()

    print("🔍 Checking environment configuration...")
    print()

    # Check required variables
    required_vars = {
        "OPENAI_API_KEY": "OpenAI API key for LLM functionality",
        "OPENAI_MODEL": "OpenAI model to use (default: gpt-4o-mini)",
        "LLM_TEMPERATURE": "LLM temperature setting (default: 0.7)",
        "LLM_MAX_TOKENS": "Maximum tokens per LLM request (default: 100)",
        "MESSAGE_FREQUENCY": "Message generation frequency (default: 0.3)",
        "COLLUSION_CONFIDENCE_THRESHOLD": "Collusion detection threshold (default: 0.7)",
    }

    all_good = True

    for var, description in required_vars.items():
        value = os.getenv(var)
        if value:
            if var == "OPENAI_API_KEY" and value == "your_openai_api_key_here":
                print(f"⚠️  {var}: {description} (needs to be set to actual key)")
                all_good = False
            else:
                print(f"✅ {var}: {value}")
        else:
            print(f"❌ {var}: {description} (not set)")
            all_good = False

    print()

    if all_good:
        print("🎉 Environment is properly configured!")
        print("🚀 You can now run: python scripts/real_llm_messages.py")
    else:
        print("⚠️  Environment needs configuration.")
        print("📝 Edit your .env file with the correct values.")


if __name__ == "__main__":
    import sys

    if len(sys.argv) > 1 and sys.argv[1] == "check":
        check_env_setup()
    else:
        create_env_file()
        print()
        check_env_setup()
