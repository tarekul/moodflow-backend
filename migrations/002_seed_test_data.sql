-- Migration: Seed test data
-- Created: 2024-10-31
-- Description: Add sample users and logs for testing

-- Insert test users
INSERT INTO users (email, password_hash) VALUES
('alice@example.com', 'temp_hash_alice'),
('bob@example.com', 'temp_hash_bob')
ON CONFLICT (email) DO NOTHING;

-- Insert specific test logs for Alice using the provided data
INSERT INTO daily_logs (
    user_id, 
    log_date, 
    mood, 
    productivity, 
    sleep_hours, 
    sleep_quality, 
    stress,
    physical_activity_min, 
    screen_time_hours, 
    diet_quality,
    social_interaction_hours, 
    weather
)
VALUES
((SELECT id FROM users WHERE email = 'alice@example.com'), '2023-06-03', 1, 2, 6.2, 'Good', 8, 39, 6.9, 'Average', 2.6, 'Cloudy'),
((SELECT id FROM users WHERE email = 'alice@example.com'), '2023-06-10', 4, 4, 6.2, 'Fair', 7, 26, 6.7, 'Average', 1.1, 'Cloudy'),
((SELECT id FROM users WHERE email = 'alice@example.com'), '2023-06-11', 6, 7, 8.4, 'Good', 2, 4, 5.1, 'Poor', 1.9, 'Sunny'),
((SELECT id FROM users WHERE email = 'alice@example.com'), '2023-06-12', 6, 8, 7.9, 'Excellent', 6, 43, 6.4, 'Poor', 3.5, 'Sunny'),
((SELECT id FROM users WHERE email = 'alice@example.com'), '2023-06-13', 7, 9, 8.4, 'Excellent', 5, 37, 1.0, 'Good', 0.4, 'Sunny'),
((SELECT id FROM users WHERE email = 'alice@example.com'), '2023-06-16', 3, 4, 6.6, 'Poor', 6, 25, 6.6, 'Good', 0.5, 'Cloudy'),
((SELECT id FROM users WHERE email = 'alice@example.com'), '2023-06-17', 6, 7, 7.9, 'Good', 3, 14, 6.0, 'Poor', 3.4, 'Cloudy'),
((SELECT id FROM users WHERE email = 'alice@example.com'), '2023-06-18', 4, 4, 4.8, 'Good', 6, 26, 2.1, 'Good', 2.2, 'Sunny'),
((SELECT id FROM users WHERE email = 'alice@example.com'), '2023-06-19', 8, 9, 7.0, 'Poor', 3, 27, 5.5, 'Good', 1.8, 'Sunny'),
((SELECT id FROM users WHERE email = 'alice@example.com'), '2023-06-29', 3, 5, 8.2, 'Excellent', 7, 44, 6.7, 'Good', 0.0, 'Cloudy'),
((SELECT id FROM users WHERE email = 'alice@example.com'), '2023-06-30', 6, 7, 6.1, 'Good', 5, 38, 3.3, 'Poor', 1.8, 'Cloudy')
ON CONFLICT (user_id, log_date) DO NOTHING;