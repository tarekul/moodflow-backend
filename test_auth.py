from utils.auth import hash_password, verify_password, create_access_token, decode_access_token

# Test password hashing
print("=== Testing Password Hashing ===")
password = "mypassword123"
hashed = hash_password(password)
print(f"Original: {password}")
print(f"Hashed: {hashed}")
print(f"Hashes match: {verify_password(password, hashed)}")
print(f"Wrong password: {verify_password('wrongpassword', hashed)}")
print()

# Test JWT tokens
print("=== Testing JWT Tokens ===")
token = create_access_token({"sub": "alice@example.com"})
print(f"Token: {token[:50]}...")
decoded = decode_access_token(token)
print(f"Decoded: {decoded}")
print()

# Test expired/invalid token
print("=== Testing Invalid Token ===")
try:
    decode_access_token("invalid.token.here")
except Exception as e:
    print(f"Expected error: {e}")
