# Physical AI & Human Robotics - Data Model

## Overview
Describes the data structures and relationships used throughout the book's examples and implementations.

## Core Entities

### Robot State
- **Position**: x, y, z coordinates in 3D space
- **Orientation**: roll, pitch, yaw angles or quaternion
- **Velocity**: linear and angular velocity vectors
- **Sensors**: current readings from all connected sensors
- **Actuators**: current positions/states of all actuators
- **Status**: operational state, battery level, error flags

### Sensor Data
#### LiDAR Data
- **Range Data**: Array of distance measurements
- **Resolution**: Angular resolution of sensor
- **Field of View**: Horizontal and vertical FOV
- **Timestamp**: When measurement was taken

#### Depth Camera Data
- **Image**: Raw depth image
- **Resolution**: Width and height of image
- **Focal Length**: Camera focal length
- **Timestamp**: When measurement was taken

#### IMU Data
- **Acceleration**: Linear acceleration (x, y, z)
- **Angular Velocity**: Rotational velocity (x, y, z)
- **Orientation**: Current orientation estimate
- **Timestamp**: When measurement was taken

### Environment Representation
#### Occupancy Grid
- **Map Resolution**: Size of each grid cell
- **Width/Height**: Dimensions in cells
- **Origin**: World coordinates of grid origin
- **Data**: Array of occupancy probabilities

#### Point Cloud
- **Points**: Array of 3D points (x, y, z)
- **Colors**: Optional RGB values for each point
- **Normals**: Surface normal vectors
- **Timestamp**: When cloud was captured

### Action Planning
#### Navigation Goal
- **Position**: Target position (x, y, z)
- **Orientation**: Target orientation (quaternion)
- **Constraints**: Movement limitations
- **Priority**: Execution priority level

#### Manipulation Action
- **Target Object**: Object to manipulate
- **Approach Vector**: Direction to approach object
- **Grasp Pose**: Position and orientation for gripper
- **Force Limits**: Maximum forces/torques to apply

## Relationships

### Robot State ↔ Sensor Data
- One robot state corresponds to multiple sensor data streams
- Sensors continuously update robot state with new measurements

### Environment Representation ↔ Robot State
- Occupancy grids and point clouds provide environmental context
- Used for navigation and obstacle avoidance

### Action Planning ↔ Robot State
- Actions modify the expected robot state
- Current state affects feasibility of planned actions

## Data Flow
1. Sensors continuously update robot state
2. Robot state informs environment representation
3. Environment representation enables action planning
4. Executed actions modify robot state
5. Cycle repeats in continuous loop

## Serialization Format
All data models use JSON serialization with standardized schemas to ensure interoperability between simulation and real-world deployments.