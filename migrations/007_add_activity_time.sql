-- Migration: Add activity time to daily logs table
-- Created: 2025-12-18
-- Description: Add activity time to daily logs table

ALTER TABLE daily_logs
ADD COLUMN IF NOT EXISTS activity_time VARCHAR(255);
