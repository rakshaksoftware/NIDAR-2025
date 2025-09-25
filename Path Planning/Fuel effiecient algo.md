## CORE LOGIC:

# Time-based launch calculation
time_now = frame_count
rest_time = TOTAL_DRONE1_TIME - time_now
time2 = calculate_drone2_delivery_time(drone1_found)

# Launch condition: rest_time <= time2 AND we have people to deliver
if not drone2_launched and rest_time <= time2 and drone1_found:
    drone2_current_path = find_shortest_path(drone1_found, drone2_pos)
    drone2_mode = "delivering"
    drone2_launched = True
### Key Characteristics

**Calculated Launch Timing**: Drone 2 remains idle at base until the optimal launch time.

**Time Analysis**: Continuously calculates:

- Remaining time for Drone 1 to complete its mission (`rest_time`)
- Time needed for Drone 2 to deliver to all currently found people (`time2`)

**Optimized Launch**: Drone 2 launches only when the remaining time for Drone 1 is less than or equal to the time Drone 2 needs for deliveries.

**Single Delivery Phase**: Drone 2 executes one efficient delivery route rather than multiple intermittent trips.

### Advantages

- **Minimal energy consumption** as Drone 2 remains idle until needed
- **Optimized route planning** with a single efficient path to all discovered people
- **Better for extended missions** where energy conservation is critical
- **Reduced operational costs** due to limited drone usage

### Disadvantages

- **Potential delays** in delivery if people are found late in the search
- **Risk of incomplete deliveries** if time calculation is inaccurate
- **Less responsive** to new discoveries after Drone 2 has launched
