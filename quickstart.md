# Quickstart Guide - Physical AI & Human Robotics Book

## Setting up Your Environment

This guide will walk you through setting up your development environment to work with the examples in the Physical AI & Human Robotics book.

### Prerequisites

Before getting started, ensure you have the following installed:

- **ROS 2 Humble Hawksbill** (with colcon build system)
- **Python 3.9+** with pip
- **Node.js 16+** and npm/yarn
- **Docker** and Docker Compose
- **Git** version control
- **Unity Hub** (for Unity simulations) or **Gazebo Garden** (for Gazebo simulations)

### Installation Steps

1. **Clone the repository**
   ```bash
   git clone https://github.com/your-repo/physical-ai-human-robotics-book.git
   cd physical-ai-human-robotics-book
   ```

2. **Install project dependencies**
   ```bash
   # Install Node.js dependencies for documentation
   npm install
   
   # Install Python dependencies
   python -m pip install --upgrade pip
   pip install -r requirements.txt
   ```

3. **Set up ROS 2 workspace**
   ```bash
   mkdir -p ~/physical_ai_ws/src
   cd ~/physical_ai_ws
   # Copy ROS packages from the book's src/ros_packages directory
   colcon build
   source install/setup.bash
   ```

4. **Configure environment variables**
   ```bash
   # Add to your bashrc/zshrc:
   export BOOK_WS=~/physical_ai_ws
   export GAZEBO_MODEL_PATH=$GAZEBO_MODEL_PATH:~/physical_ai_ws/src/models
   export ROS_PACKAGE_PATH=$ROS_PACKAGE_PATH:~/physical_ai_ws/src
   ```

### Running Examples by Chapter

#### Chapter 1: Physical AI Foundations
```bash
# Navigate to Chapter 1 examples
cd ~/physical_ai_ws/src/chapter1_examples

# Run sensor simulation
ros2 launch sensor_simulation.launch.py

# Run motor control demo
ros2 run motor_control_demo basic_control
```

#### Chapter 2: Digital Twin Simulation
```bash
# Launch Gazebo environment
ros2 launch gazebo_env.launch.py world:=simple_room

# Or use Unity environment (if available)
./unity_simulation_executable --scene basic_environment
```

#### Chapter 3: AI-Robot Brain
```bash
# Start perception nodes
ros2 launch perception_stack.launch.py

# Run VSLAM algorithm
ros2 run isaac_ros_vslam vslam_node
```

#### Chapter 4: Vision-Language-Action
```bash
# Launch voice input pipeline
ros2 launch whisper_voice_pipeline.launch.py

# Start LLM planning interface
python llm_planning_interface.py
```

#### Chapter 5: Capstone - Autonomous Humanoid
```bash
# Launch complete system
ros2 launch full_humanoid_system.launch.py

# Send voice command
echo "Move to kitchen and pick up the red cup" | ros2 topic pub /voice_input std_msgs/String
```

### Documentation Website

To run the documentation website locally:

```bash
# Install dependencies if not done already
npm install

# Start development server
npm start
```

Visit `http://localhost:3000` to view the documentation in your browser.

### Troubleshooting Common Issues

**Issue**: ROS 2 nodes not communicating across terminals
- **Solution**: Make sure to source the ROS environment in each terminal:
  ```bash
  source ~/physical_ai_ws/install/setup.bash
  ```

**Issue**: Gazebo not launching properly
- **Solution**: Check if GPU drivers are properly installed and run:
  ```bash
  export LIBGL_ALWAYS_SOFTWARE=1  # As fallback for rendering issues
  ```

**Issue**: Python modules not found
- **Solution**: Ensure you're running in the correct virtual environment:
  ```bash
  python -c "import sys; print(sys.path)"  # Check your Python path
  ```

### Repository Structure

For easy navigation of the codebase:

```
physical-ai-human-robotics-book/
├── docs/              # Documentation source files
├── src/               # Source code for examples
│   ├── simulation/    # Simulation code
│   ├── perception/    # Perception algorithms
│   ├── navigation/    # Navigation code
│   └── manipulation/  # Manipulation routines
├── assets/            # Images, videos, diagrams
├── book/              # Book chapters content
├── tests/             # Unit and integration tests
└── scripts/           # Utility scripts
```

That's it! You're now ready to follow along with the examples in the Physical AI & Human Robotics book.