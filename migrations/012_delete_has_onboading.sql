-- Migration 012_delete_has_onboading.sql
-- Created: 2026-01-02
-- Description: Delete has_onboarded column from users table

ALTER TABLE users DROP COLUMN has_onboarded;
