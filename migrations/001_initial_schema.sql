-- Migration: Initial Schema
-- Created: 2024-10-31
-- Description: Create users and daily_logs tables

-- Create users table
CREATE TABLE IF NOT EXISTS users (
    id SERIAL PRIMARY KEY,
    email VARCHAR(255) UNIQUE NOT NULL,
    password_hash VARCHAR(255) NOT NULL,
    tier VARCHAR(20) DEFAULT 'free',
    trial_ends_at TIMESTAMP,
    created_at TIMESTAMP DEFAULT NOW()
);

-- Create daily_logs table
CREATE TABLE IF NOT EXISTS daily_logs (
    id SERIAL PRIMARY KEY,
    user_id INTEGER REFERENCES users(id) ON DELETE CASCADE,
    log_date DATE NOT NULL,
    mood FLOAT CHECK (mood >= 1 AND mood <= 10),
    productivity FLOAT CHECK (productivity >= 1 AND productivity <= 10),
    sleep_hours FLOAT CHECK (sleep_hours >= 0 AND sleep_hours <= 24),
    sleep_quality VARCHAR(20),
    stress FLOAT CHECK (stress >= 1 AND stress <= 10),
    physical_activity_min INTEGER CHECK (physical_activity_min >= 0),
    screen_time_hours FLOAT CHECK (screen_time_hours >= 0),
    diet_quality VARCHAR(20),
    social_interaction_hours FLOAT CHECK (social_interaction_hours >= 0),
    weather VARCHAR(20),
    notes TEXT,
    created_at TIMESTAMP DEFAULT NOW(),
    UNIQUE(user_id, log_date)
);

-- Create indexes for better query performance
CREATE INDEX IF NOT EXISTS idx_daily_logs_user_id ON daily_logs(user_id);
CREATE INDEX IF NOT EXISTS idx_daily_logs_date ON daily_logs(log_date);