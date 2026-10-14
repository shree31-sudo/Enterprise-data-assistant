#!/usr/bin/env python3
"""
Debug script to test LLM response and see raw output
Run this to verify Qwen is returning proper JSON format
"""

import json
import requests
import os
import sys
from pathlib import Path

# Configuration
LLM_BASE_URL = os.getenv("LLM_BASE_URL", "http://10.225.30.26:11434").rstrip("/")
LLM_MODEL = os.getenv("LLM_MODEL", "qwen3:14b")
LLM_CONNECT_TIMEOUT = float(os.getenv("LLM_CONNECT_TIMEOUT", "10"))
LLM_TIMEOUT = float(os.getenv("LLM_TIMEOUT", "1800"))
LLM_TEMPERATURE = float(os.getenv("LLM_TEMPERATURE", "0.1"))

# Simple test prompt
TEST_PROMPT = """You are a JSON classifier.
CRITICAL: Output ONLY valid JSON. NO text before or after.
NO markdown fences. NO EXPLANATION.
Start with { immediately.

Classify this single table as JSON:
{
  "MFC_M_BOM": {
    "table_type": "MASTER",
    "business_purpose": "Stores master data for bill of materials",
    "columns": {
      "BOM_ID": {
        "description": "Unique system-generated ID",
        "is_time_column": false,
        "is_status_column": false
      },
      "Cuser": {
        "description": "User who created the BOM record",
        "is_time_column": false,
        "is_status_column": false
      }
    }
  }
}

Output ONLY this JSON format:
{
  "MFC_M_BOM": {
    "sensitivity_assessment": "reason",
    "columns": {
      "BOM_ID": {
        "sensitive": false,
        "reason": "Primary key"
      },
      "Cuser": {
        "sensitive": true,
        "reason": "User audit field"
      }
    }
  }
}
"""

def test_llm_response():
    """Test LLM response without parsing, just show raw output"""
    print("\n" + "=" * 80)
    print("LLM RESPONSE DEBUG TEST")
    print("=" * 80)
    
    print(f"\n📡 Configuration:")
    print(f"   URL: {LLM_BASE_URL}")
    print(f"   Model: {LLM_MODEL}")
    print(f"   Temperature: {LLM_TEMPERATURE}")
    print(f"   Timeout: {LLM_TIMEOUT}s")
    
    print(f"\n🚀 Sending test prompt...")
    
    try:
        response = requests.post(
            f"{LLM_BASE_URL}/api/generate",
            json={
                "model": LLM_MODEL,
                "prompt": TEST_PROMPT,
                "stream": False,
                "temperature": LLM_TEMPERATURE,
            },
            timeout=(LLM_CONNECT_TIMEOUT, LLM_TIMEOUT)
        )
        
        if response.status_code != 200:
            print(f"❌ Error: {response.status_code}")
            print(f"   {response.text}")
            return False
        
        result = response.json()
        raw_response = result.get('response', '')
        
        print(f"\n✅ Got response ({len(raw_response)} chars)")
        
        print(f"\n📋 RAW RESPONSE:")
        print("=" * 80)
        print(raw_response)
        print("=" * 80)
        
        # Try to find JSON
        print(f"\n🔍 Analysis:")
        print(f"   Starts with '{{': {raw_response.strip().startswith('{')}")
        print(f"   Ends with '}}': {raw_response.strip().endswith('}')}")
        
        # Try to extract JSON
        cleaned = raw_response.strip()
        start_idx = cleaned.find("{")
        end_idx = cleaned.rfind("}")
        
        if start_idx != -1 and end_idx != -1:
            json_part = cleaned[start_idx:end_idx+1]
            print(f"   JSON part found: {len(json_part)} chars")
            
            try:
                parsed = json.loads(json_part)
                print(f"   ✅ Valid JSON!")
                print(f"\n   Parsed structure:")
                print(json.dumps(parsed, indent=2)[:500])
                return True
            except json.JSONDecodeError as e:
                print(f"   ❌ Invalid JSON: {e}")
                print(f"   Problem area: {json_part[max(0, e.pos-50):e.pos+50]}")
                return False
        else:
            print(f"   ❌ No JSON braces found!")
            return False
    
    except requests.exceptions.ConnectionError:
        print(f"❌ Cannot connect to {LLM_BASE_URL}")
        return False
    except Exception as e:
        print(f"❌ Error: {e}")
        return False

if __name__ == "__main__":
    success = test_llm_response()
    sys.exit(0 if success else 1)
