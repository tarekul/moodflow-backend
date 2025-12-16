-- Migration: Add first and last name to users table
-- Created: 2025-12-15
-- Description: Add first and last name to users table

-- Add first and last name columns
ALTER TABLE users
ADD COLUMN IF NOT EXISTS first_name VARCHAR(255),
ADD COLUMN IF NOT EXISTS last_name VARCHAR(255);
