## CORE LOGIC:
# Drone 2 starts immediately and follows Drone 1
drone2_mode = "following"  # Start by following drone 1

# When people are found, Drone 2 switches to delivery mode
if drone1_found and any(p not in drone2_delivered for p in drone1_found):
    drone2_current_path = find_shortest_path(undelivered, drone2_pos)
    drone2_mode = "delivering"
    
### Key Characteristics

**Simultaneous Operation**: Both drones begin operations immediately at simulation start.

**Following Behavior**: Drone 2 follows Drone 1 at a fixed offset (30 pixels), maintaining proximity to newly discovered people.

**Reactive Delivery**: Drone 2 immediately delivers to any person found by Drone 1, minimizing response time.

**Continuous Operation**: Drone 2 works throughout the entire mission duration rather than waiting for an optimal launch time.

### Advantages

- **Minimal response time** between discovery and delivery
- **Higher probability** of completing all deliveries before Drone 1 finishes searching
- **Better for time-critical** missions where prompt delivery is essential

### Disadvantages

- **Higher energy consumption** as Drone 2 is active throughout the mission
- **Less fuel-efficient** due to continuous operation and following behavior
- **Potential overkill** if many people are found early in the search
