-- Migration: Add tags to daily logs table
-- Created: 2025-12-20
-- Description: Add tags column to daily_logs table

ALTER TABLE daily_logs ADD COLUMN tags TEXT[] DEFAULT NULL;