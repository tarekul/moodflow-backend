-- Migration 013_drop_diet_quality.sql
-- Created: 2026-01-04
-- Description: Delete diet_quality column from daily_logs table

ALTER TABLE daily_logs DROP COLUMN diet_quality;