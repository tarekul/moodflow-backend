-- Migration: Add Password Reset Columns
-- Created: 2025-12-11
-- Description: Add reset_token and reset_token_expires_at to users table

-- Add password reset columns
ALTER TABLE users
ADD COLUMN IF NOT EXISTS reset_token VARCHAR(255),
ADD COLUMN IF NOT EXISTS reset_token_expires_at TIMESTAMP;

-- Create index for faster token lookups
CREATE INDEX IF NOT EXISTS idx_users_reset_token ON users(reset_token);