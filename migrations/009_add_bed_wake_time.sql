-- Migration: Add sleep bed and wake time to logs table
-- Created: 2025-12-24
-- Description: Add sleep_bed_time and sleep_wake_time columns to logs table

ALTER TABLE daily_logs
ADD COLUMN sleep_bed_time TIME,
ADD COLUMN sleep_wake_time TIME;