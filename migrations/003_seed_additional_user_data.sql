-- Migration: Seed additional user data
-- Created: 2025-11-12
-- Description: Add additional sample users and logs for testing

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
((SELECT id FROM users WHERE email = 'bob@example.com'), '2023-06-02', 6, 6, 8, 'Good', 4, 0, 4.8, 'Average', 3.4, 'Cloudy'),
((SELECT id FROM users WHERE email = 'bob@example.com'), '2023-06-09', 4, 7, 10, 'Excellent', 6, 39, 4, 'Average', 3.4, 'Sunny'),
((SELECT id FROM users WHERE email = 'bob@example.com'), '2023-06-14', 8, 9, 6.5, 'Fair', 1, 34, 4.4, 'Good', 1.4, 'Sunny'),
((SELECT id FROM users WHERE email = 'bob@example.com'), '2023-06-15', 10, 10, 8, 'Fair', 1, 62, 4, 'Poor', 0, 'Cloudy'),
((SELECT id FROM users WHERE email = 'bob@example.com'), '2023-06-21', 6, 7, 7.2, 'Poor', 4, 14, 4.5, 'Good', 1.7, 'Sunny'),
((SELECT id FROM users WHERE email = 'bob@example.com'), '2023-06-22', 2, 4, 7.5, 'Good', 9, 40, 5.7, 'Good', 2.7, 'Rainy'),
((SELECT id FROM users WHERE email = 'bob@example.com'), '2023-06-27', 4, 7, 9.6, 'Good', 6, 48, 5.3, 'Average', 3.2, 'Rainy'),
((SELECT id FROM users WHERE email = 'charlie@example.com'), '2023-06-07', 8, 9, 8, 'Excellent', 3, 20, 7, 'Good', 1.7, 'Rainy'),
((SELECT id FROM users WHERE email = 'charlie@example.com'), '2023-06-08', 10, 10, 9.7, 'Good', 1, 34, 3.2, 'Poor', 1.1, 'Sunny'),
((SELECT id FROM users WHERE email = 'charlie@example.com'), '2023-06-10', 7, 9, 8, 'Fair', 4, 35, 6.8, 'Average', 3.9, 'Sunny'),
((SELECT id FROM users WHERE email = 'charlie@example.com'), '2023-06-18', 10, 10, 6.3, 'Good', 1, 7, 1.9, 'Good', 2.9, 'Rainy'),
((SELECT id FROM users WHERE email = 'charlie@example.com'), '2023-06-21', 5, 6, 5.5, 'Fair', 4, 40, 4.3, 'Average', 4.8, 'Sunny'),
((SELECT id FROM users WHERE email = 'charlie@example.com'), '2023-06-22', 5, 5, 6, 'Good', 6, 21, 6.2, 'Average', 2.6, 'Cloudy'),
((SELECT id FROM users WHERE email = 'charlie@example.com'), '2023-06-23', 9, 10, 9.9, 'Good', 2, 33, 6.4, 'Good', 2.8, 'Sunny'),
((SELECT id FROM users WHERE email = 'charlie@example.com'), '2023-06-26', 7, 7, 7.4, 'Good', 3, 22, 8.7, 'Good', 1.7, 'Sunny'),
((SELECT id FROM users WHERE email = 'charlie@example.com'), '2023-06-27', 2, 2, 5, 'Good', 9, 33, 5.6, 'Good', 4.2, 'Rainy'),
((SELECT id FROM users WHERE email = 'charlie@example.com'), '2023-06-28', 5, 6, 8.7, 'Poor', 6, 12, 2.9, 'Good', 0.8, 'Rainy');

