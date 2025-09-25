import pygame
import random
import math

# ----------------- Pygame setup -----------------
pygame.init()
WIDTH, HEIGHT = 800, 600
WIN = pygame.display.set_mode((WIDTH, HEIGHT))
pygame.display.set_caption("Drone Time-Based Launch Simulation")
FONT = pygame.font.SysFont("Arial", 18)
SMALL_FONT = pygame.font.SysFont("Arial", 12)
clock = pygame.time.Clock()

# ----------------- Colors -----------------
WHITE = (255, 255, 255)
BLACK = (0, 0, 0)
RED   = (255, 0, 0)
BLUE  = (0, 100, 255)
ORANGE = (255, 165, 0)
GREEN = (0, 255, 0)
GRAY = (128, 128, 128)
LIGHT_GRAY = (200, 200, 200)
YELLOW = (255, 255, 0)
PURPLE = (128, 0, 128)

# ----------------- Grid and Launch Pad Setup -----------------
GRID_SIZE = 100  # Larger grid squares
DRONE1_START = [60, 60]  # Drone 1 start position (left side)
DRONE2_START = [120, 60]  # Drone 2 start position (right side)

# Calculate grid to fit within boundaries with margins
MARGIN = 80
available_width = WIDTH - 2 * MARGIN
available_height = HEIGHT - 2 * MARGIN

cols = available_width // GRID_SIZE
rows = available_height // GRID_SIZE

# Center the grid
grid_start_x = MARGIN + (available_width - cols * GRID_SIZE) // 2
grid_start_y = MARGIN + (available_height - rows * GRID_SIZE) // 2

# Generate S-shaped path through grid centers
def generate_s_path_centers():
    centers = []
    for row in range(rows):
        if row % 2 == 0:  # Left to right (even rows)
            for col in range(cols):
                x = grid_start_x + col * GRID_SIZE + GRID_SIZE // 2
                y = grid_start_y + row * GRID_SIZE + GRID_SIZE // 2
                centers.append((x, y))
        else:  # Right to left (odd rows)
            for col in range(cols-1, -1, -1):
                x = grid_start_x + col * GRID_SIZE + GRID_SIZE // 2
                y = grid_start_y + row * GRID_SIZE + GRID_SIZE // 2
                centers.append((x, y))
    return centers

S_PATH_CENTERS = generate_s_path_centers()

# Generate people ONLY within the grid boundaries
PEOPLE = []
for _ in range(10):
    # Ensure people are within grid bounds
    x = random.randint(grid_start_x + 10, grid_start_x + cols * GRID_SIZE - 10)
    y = random.randint(grid_start_y + 10, grid_start_y + rows * GRID_SIZE - 10)
    PEOPLE.append((x, y))

# Detection range = distance from center to corner of grid square
detection_range = math.sqrt((GRID_SIZE/2)**2 + (GRID_SIZE/2)**2)

# Drone positions and states
drone1_pos = list(DRONE1_START)
drone2_pos = list(DRONE2_START)

# Speeds
drone1_speed = 6
drone2_speed = 1

# Time configuration variables
DRONE1_SCAN_TIME = 90  # frames drone 1 spends scanning at each grid center (1.5 seconds at 60 FPS)
DRONE2_DELIVERY_TIME = 120  # frames drone 2 spends delivering at each person (2 seconds at 60 FPS)

# Drone 1 variables
current_center_index = 0
drone1_at_center = False
drone1_scan_timer = 0
drone1_completed = False
drone1_returning = False
drone1_at_base = True

# Drone 2 variables - MODIFIED FOR NEW BEHAVIOR
drone2_mode = "following"  # Start by following drone 1
drone2_delivered = set()  # Use set for faster lookup
drone2_current_path = []
drone2_path_index = 0
drone2_delivery_timer = 0
drone2_at_target = False
drone2_completed = False
drone2_at_base = False  # Start at base but will immediately begin following
drone2_launched = True  # Start launched from the beginning

# Time tracking
frame_count = 0
FPS = 60

# Found people
drone1_found = []

# ----------------- Helper Functions -----------------
def distance(a, b):
    return math.sqrt((a[0]-b[0])**2 + (a[1]-b[1])**2)

def move_towards(pos, target, speed):
    if not target:
        return pos
    dx, dy = target[0] - pos[0], target[1] - pos[1]
    dist = distance(pos, target)
    if dist <= speed:
        return list(target)
    return [pos[0] + (dx/dist) * speed, pos[1] + (dy/dist) * speed]

def find_shortest_path(targets, start_pos):
    """Simple nearest neighbor path - no complex optimization"""
    if not targets:
        return []
    
    # Always use simple nearest neighbor algorithm
    unvisited = targets[:]
    path = []
    current = start_pos
    
    while unvisited:
        # Find nearest unvisited target
        nearest = min(unvisited, key=lambda p: distance(current, p))
        path.append(nearest)
        current = nearest
        unvisited.remove(nearest)
    
    return path

def calculate_drone1_total_time():
    """Calculate total time for Drone 1 to complete full path (travel + detection)"""
    # Time to move between all grid centers
    total_movement_time = 0
    for i in range(len(S_PATH_CENTERS) - 1):
        dist = distance(S_PATH_CENTERS[i], S_PATH_CENTERS[i+1])
        total_movement_time += dist / drone1_speed
    
    # Time to scan at each center
    total_scan_time = len(S_PATH_CENTERS) * DRONE1_SCAN_TIME
    
    # Time to return to base
    return_time = distance(S_PATH_CENTERS[-1], DRONE1_START) / drone1_speed
    
    return total_movement_time + total_scan_time + return_time

def calculate_drone2_delivery_time(human_locations):
    """Calculate minimum time Drone-2 needs to deliver to all humans"""
    if not human_locations:
        return 0
    
    # Find shortest path to visit all humans
    path = find_shortest_path(human_locations, DRONE2_START)
    
    # Calculate travel time
    total_travel_time = 0
    current_pos = DRONE2_START
    
    for target in path:
        dist = distance(current_pos, target)
        total_travel_time += dist / drone2_speed
        current_pos = target
    
    # Add delivery time for each person
    total_delivery_time = len(human_locations) * DRONE2_DELIVERY_TIME
    
    return total_travel_time + total_delivery_time

def draw_grid():
    """Draw the search grid"""
    for row in range(rows):
        for col in range(cols):
            x = grid_start_x + col * GRID_SIZE
            y = grid_start_y + row * GRID_SIZE
            rect = pygame.Rect(x, y, GRID_SIZE, GRID_SIZE)
            pygame.draw.rect(WIN, LIGHT_GRAY, rect, 1)
    
    # Draw S-path centers
    for i, center in enumerate(S_PATH_CENTERS):
        if i < current_center_index:
            color = GREEN
        elif i == current_center_index:
            color = YELLOW
        else:
            color = GRAY
        pygame.draw.circle(WIN, color, center, 4)

# Pre-calculate total time for Drone 1
TOTAL_DRONE1_TIME = calculate_drone1_total_time()

# ----------------- Main Loop -----------------
running = True
while running:
    clock.tick(FPS)
    WIN.fill(WHITE)
    frame_count += 1

    # ----------------- Events -----------------
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False

    # ----------------- Draw Grid -----------------
    draw_grid()

    # ----------------- Drone 1: S-Path Grid Search -----------------
    if not drone1_completed:
        # Check if all people have been found - if so, return to base immediately
        if len(drone1_found) >= 10 and not drone1_returning:
            drone1_returning = True
        
        if drone1_returning:
            # Return to base
            drone1_pos = move_towards(drone1_pos, DRONE1_START, drone1_speed)
            
            if distance(drone1_pos, DRONE1_START) < 5:
                drone1_completed = True
                drone1_at_base = True
                
        elif current_center_index < len(S_PATH_CENTERS):
            target_center = S_PATH_CENTERS[current_center_index]
            
            # Move to center if not there yet
            if not drone1_at_center:
                drone1_at_base = False
                drone1_pos = move_towards(drone1_pos, target_center, drone1_speed)
                
                # Check if reached center (with tolerance)
                if distance(drone1_pos, target_center) < 5:
                    drone1_pos = list(target_center)
                    drone1_at_center = True
                    drone1_scan_timer = DRONE1_SCAN_TIME
            
            # Scan at center
            else:
                # Detect people within range
                for person in PEOPLE:
                    if person not in drone1_found and distance(drone1_pos, person) <= detection_range:
                        drone1_found.append(person)
                
                drone1_scan_timer -= 1
                
                # Move to next center when scan complete
                if drone1_scan_timer <= 0:
                    current_center_index += 1
                    drone1_at_center = False
        
        else:
            # All centers visited, return to base
            drone1_returning = True
            drone1_pos = move_towards(drone1_pos, DRONE1_START, drone1_speed)
            
            if distance(drone1_pos, DRONE1_START) < 5:
                drone1_completed = True
                drone1_at_base = True

    # ----------------- Drone 2: NEW BEHAVIOR -----------------
    # Drone 2 starts at the same time as Drone 1 and follows it
    # Only delivers to people when they are found and returns to base only after all deliveries
    
    if drone2_mode == "following":
        # Follow Drone 1 at a slight distance
        follow_offset = 30  # Distance to maintain from Drone 1
        target_x = drone1_pos[0] + follow_offset
        target_y = drone1_pos[1] + follow_offset
        
        # Move towards the follow position
        drone2_pos = move_towards(drone2_pos, (target_x, target_y), drone2_speed)
        
        # Check if we have people to deliver to
        if drone1_found and any(p not in drone2_delivered for p in drone1_found):
            # Switch to delivery mode if there are undelivered people
            undelivered = [p for p in drone1_found if p not in drone2_delivered]
            drone2_current_path = find_shortest_path(undelivered, drone2_pos)
            drone2_mode = "delivering"
            drone2_path_index = 0
            drone2_at_target = False
    
    elif drone2_mode == "delivering":
        if drone2_delivery_timer > 0:
            # Currently making delivery
            drone2_delivery_timer -= 1
            if drone2_delivery_timer == 0:
                # Delivery complete
                if drone2_path_index > 0:
                    delivered_person = drone2_current_path[drone2_path_index - 1]
                    drone2_delivered.add(delivered_person)
                
                # Reset target flags to continue to next target
                drone2_at_target = False
                
                # Check if all people have been delivered
                if len(drone2_delivered) >= 10:
                    drone2_mode = "returning"
        
        elif drone2_path_index < len(drone2_current_path):
            # Move to next target
            target = drone2_current_path[drone2_path_index]
            
            if not drone2_at_target:
                drone2_pos = move_towards(drone2_pos, target, drone2_speed)
                
                # Check if reached target
                if distance(drone2_pos, target) < 5:
                    drone2_pos = list(target)
                    drone2_at_target = True
                    drone2_delivery_timer = DRONE2_DELIVERY_TIME
                    drone2_path_index += 1  # Move to next target index
        
        else:
            # Finished current path - check for more targets
            remaining_undelivered = [p for p in drone1_found if p not in drone2_delivered]
            if remaining_undelivered and len(drone2_delivered) < 10:
                # More targets available, create new path
                drone2_current_path = find_shortest_path(remaining_undelivered, drone2_pos)
                drone2_path_index = 0
                drone2_at_target = False
            else:
                # No more targets to deliver to, go back to following
                drone2_mode = "following"
    
    elif drone2_mode == "returning":
        # Return to base after all deliveries are complete
        drone2_pos = move_towards(drone2_pos, DRONE2_START, drone2_speed)
        
        if distance(drone2_pos, DRONE2_START) < 5:
            drone2_completed = True
            drone2_at_base = True

    # ----------------- Draw People -----------------
    for i, person in enumerate(PEOPLE):
        # Color coding based on status
        if person in drone2_delivered:
            color = BLUE  # Delivered
        elif person in drone1_found:
            color = GREEN  # Found but not delivered
        else:
            color = RED   # Not found yet
        
        pygame.draw.circle(WIN, color, person, 8)
        WIN.blit(SMALL_FONT.render(f"P{i+1}", True, BLACK), (person[0]+10, person[1]-10))

    # ----------------- Draw Drones -----------------
    # Drone 1
    if drone1_completed:
        d1_color = GREEN
    elif drone1_returning:
        d1_color = ORANGE
    else:
        d1_color = RED
    
    pygame.draw.circle(WIN, d1_color, (int(drone1_pos[0]), int(drone1_pos[1])), 12)
    WIN.blit(SMALL_FONT.render("D1", True, WHITE), (int(drone1_pos[0])-8, int(drone1_pos[1])-6))
    
    # Show detection range when scanning
    if drone1_at_center and drone1_scan_timer > 0:
        pygame.draw.circle(WIN, (255, 0, 0, 100), (int(drone1_pos[0]), int(drone1_pos[1])), 
                          int(detection_range), 2)
    
    # Drone 2
    if drone2_completed:
        d2_color = GREEN
    elif drone2_mode == "following":
        d2_color = PURPLE  # Different color for following mode
    elif drone2_mode == "delivering":
        d2_color = BLUE
    else:  # returning
        d2_color = ORANGE
    
    pygame.draw.circle(WIN, d2_color, (int(drone2_pos[0]), int(drone2_pos[1])), 12)
    WIN.blit(SMALL_FONT.render("D2", True, WHITE), (int(drone2_pos[0])-8, int(drone2_pos[1])-6))

    # ----------------- Draw Paths -----------------
    # Draw S-path for Drone 1
    if S_PATH_CENTERS:
        for i in range(len(S_PATH_CENTERS)-1):
            start = S_PATH_CENTERS[i]
            end = S_PATH_CENTERS[i+1]
            color = GREEN if i < current_center_index else GRAY
            pygame.draw.line(WIN, color, start, end, 2)
    
    # Draw Drone 2's planned path
    if drone2_mode == "delivering" and drone2_current_path and drone2_path_index < len(drone2_current_path):
        remaining_path = drone2_current_path[drone2_path_index:]
        if remaining_path:
            # Line from current position to first remaining target
            pygame.draw.line(WIN, BLUE, drone2_pos, remaining_path[0], 2)
            
            # Lines between remaining targets
            for i in range(len(remaining_path)-1):
                pygame.draw.line(WIN, BLUE, remaining_path[i], remaining_path[i+1], 2)

    # ----------------- Draw Launch Pads -----------------
    # Drone 1 launch pad
    pygame.draw.rect(WIN, RED, (DRONE1_START[0]-20, DRONE1_START[1]-20, 40, 40), 3)
    WIN.blit(SMALL_FONT.render("D1 BASE", True, RED), (DRONE1_START[0]-25, DRONE1_START[1]+25))
    
    # Drone 2 launch pad
    pygame.draw.rect(WIN, BLUE, (DRONE2_START[0]-20, DRONE2_START[1]-20, 40, 40), 3)
    WIN.blit(SMALL_FONT.render("D2 BASE", True, BLUE), (DRONE2_START[0]-25, DRONE2_START[1]+25))

    # ----------------- Status Display -----------------
    status_x, status_y = 10, HEIGHT - 180
    
    WIN.blit(FONT.render(f"Found: {len(drone1_found)}/10", True, BLACK), (status_x, status_y))
    WIN.blit(FONT.render(f"Delivered: {len(drone2_delivered)}/10", True, BLACK), (status_x, status_y + 20))
    
    # Drone status
    if drone1_completed:
        d1_status = "At Base" if drone1_at_base else "Completed"
    elif drone1_returning:
        d1_status = "Returning"
    else:
        d1_status = "Searching"
    
    WIN.blit(FONT.render(f"Drone 1: {d1_status}", True, BLACK), (status_x, status_y + 40))
    
    if drone2_completed:
        d2_status = "At Base" if drone2_at_base else "Completed"
    else:
        d2_status = drone2_mode.title()
    
    WIN.blit(FONT.render(f"Drone 2: {d2_status}", True, BLACK), (status_x, status_y + 60))
    
    # Grid progress
    if not drone1_completed and not drone1_returning:
        progress = f"{current_center_index}/{len(S_PATH_CENTERS)}"
        WIN.blit(FONT.render(f"Grid: {progress}", True, BLACK), (status_x, status_y + 80))
    elif drone1_returning:
        WIN.blit(FONT.render(f"Grid: Early return!", True, ORANGE), (status_x, status_y + 80))
    
    # Mission status
    if drone1_completed and drone2_completed:
        WIN.blit(FONT.render("MISSION COMPLETE - ALL DRONES AT BASE", True, GREEN), (WIDTH//2 - 150, 30))
    elif len(drone2_delivered) >= 10:
        WIN.blit(FONT.render("ALL PEOPLE RESCUED!", True, GREEN), (WIDTH//2 - 80, 30))

    # Time configuration display
    time_config_x = WIDTH - 180
    time_config_y = 10
    WIN.blit(SMALL_FONT.render("Time Configuration:", True, BLACK), (time_config_x, time_config_y))
    WIN.blit(SMALL_FONT.render(f"D1 scan time: {DRONE1_SCAN_TIME/FPS:.1f}s", True, BLACK), 
             (time_config_x, time_config_y + 15))
    WIN.blit(SMALL_FONT.render(f"D2 delivery time: {DRONE2_DELIVERY_TIME/FPS:.1f}s", True, BLACK), 
             (time_config_x, time_config_y + 30))
    
    # Detection range info
    WIN.blit(SMALL_FONT.render(f"Detection range: {int(detection_range)}", True, BLACK), (WIDTH - 180, HEIGHT - 60))
    
    # Legend
    legend_x = WIDTH - 180
    legend_y = HEIGHT - 140
    WIN.blit(SMALL_FONT.render("People:", True, BLACK), (legend_x, legend_y))
    WIN.blit(SMALL_FONT.render("● Red = Undiscovered", True, RED), (legend_x, legend_y + 15))
    WIN.blit(SMALL_FONT.render("● Green = Found", True, GREEN), (legend_x, legend_y + 30))
    WIN.blit(SMALL_FONT.render("● Blue = Delivered", True, BLUE), (legend_x, legend_y + 45))
    
    # Drone 2 status legend
    WIN.blit(SMALL_FONT.render("Drone 2:", True, BLACK), (legend_x, legend_y + 60))
    WIN.blit(SMALL_FONT.render("● Purple = Following", True, PURPLE), (legend_x, legend_y + 75))
    WIN.blit(SMALL_FONT.render("● Blue = Delivering", True, BLUE), (legend_x, legend_y + 90))
    WIN.blit(SMALL_FONT.render("● Orange = Returning", True, ORANGE), (legend_x, legend_y + 105))

    pygame.display.update()

pygame.quit()
