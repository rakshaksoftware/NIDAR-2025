Two drone simulation algorithms demonstrating different coordination strategies between a search drone (Drone 1) and a delivery drone (Drone 2). 
The first algorithm prioritizes **time efficiency** while the second focuses on **fuel efficiency**.

## Shared Components

Both algorithms share these core components:
1. Grid system
2. Path finding

### Visualization System

Both implementations feature comprehensive visualization including:

- Grid system with S-path progression
- People with color-coded status (undiscovered, found, delivered)
- Drone paths and detection ranges
- Status panels with mission progress
- Time configuration displays
- Legend for interpreting visual elements

## Performance Comparison

| Aspect | Time-Efficient Algorithm | Fuel-Efficient Algorithm |
| --- | --- | --- |
| **Response Time** | Immediate | Delayed until calculated launch |
| **Energy Use** | High | Low |
| **Completion Rate** | High (especially early finds) | Variable (depends on timing) |
| **Operational Complexity** | Moderate (dynamic pathing) | High (time calculations) |
| **Best For** | Time-critical missions | Energy-constrained missions |
