-- Migration: Add custom_tags table
-- Created: 2025-12-30
-- Description: Add custom_tags table

CREATE TABLE custom_tags (
    id SERIAL PRIMARY KEY,
    user_id INT NOT NULL,
    tag_name VARCHAR(255) NOT NULL,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);