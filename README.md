# ObjectDetector

This project utilizes ORB feature detection and description to compute the key points of an object alone, and in a scene. It then uses a Brute-Force comparison to match the features. From there it estimates the object's transformation in the scene image using a similarity matrix and a RANSAC framework to estimate the projective transform.
