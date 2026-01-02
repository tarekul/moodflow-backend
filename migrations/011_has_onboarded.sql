-- Migration 011_has_onboarded.sql
-- Created: 2025-12-31
-- Description: Add has_onboarded column to users table


ALTER TABLE users ADD COLUMN has_onboarded BOOLEAN DEFAULT FALSE;