-- Migration: Add Google ID Column and drop not null from password_hash
-- Created: 2024-12-11
-- Description: Add google_id to users table

-- Add google_id column
ALTER TABLE users
ADD COLUMN IF NOT EXISTS google_id VARCHAR(255);

-- Drop not null constraint from password_hash
ALTER TABLE users
ALTER COLUMN password_hash DROP NOT NULL;

-- Create index for faster lookups
CREATE INDEX idx_users_email ON users (email);
