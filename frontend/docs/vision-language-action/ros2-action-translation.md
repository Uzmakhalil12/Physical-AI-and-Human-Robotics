---
sidebar_position: 3
---

# ROS 2 Action Translation

## Introduction to ROS 2 Action Translation

In Physical AI systems, the bridge between high-level reasoning (often performed using LLMs or other cognitive systems) and low-level robot control is critical for effective operation. ROS 2 (Robot Operating System 2) provides the communication infrastructure for this bridge through its action system, which enables long-running tasks with feedback and goal management.

Action translation involves converting high-level commands and plans into specific ROS 2 action calls that can be executed by robot components. This translation layer is essential for creating flexible, robust robotic systems that can respond to dynamic environments and high-level instructions.

## Understanding ROS 2 Actions

### Actions vs. Services vs. Topics

**Topics (Publish/Subscribe)**:
- Asynchronous, unidirectional communication
- Used for continuous data streams (sensor data, commands)
- No acknowledgment of receipt

**Services (Request/Response)**:
- Synchronous, bidirectional communication
- Used for single request-response interactions
- Request blocks until response received

**Actions (Goal/Feedback/Result)**:
- Asynchronous, stateful communication for long-running tasks
- Used for complex operations with intermediate feedback
- Supports goal preemption and status monitoring

### Action Architecture

An action consists of three message types:
1. **Goal**: Parameters sent to the action server to initiate a task
2. **Feedback**: Periodic updates on the progress of the task
3. **Result**: Final outcome of the completed task

The communication flow:
```
Action Client → Goal → Action Server
Action Server → Feedback → Action Client
Action Server → Result → Action Client
```

### Action States

Actions can be in one of several states:
- **PENDING**: Goal received but not yet started
- **ACTIVE**: Goal is being processed
- **PREEMPTING**: Goal is being canceled
- **SUCCEEDED**: Goal completed successfully
- **ABORTED**: Goal execution failed
- **CANCELED**: Goal was canceled

## Action Translation Framework

### The Translation Process

The action translation process converts high-level commands to ROS 2 actions through several steps:

```python
class ActionTranslator:
    def __init__(self):
        self.action_interfaces = {}
        self.semantic_mapping = {}
        self.context_manager = ContextManager()
    
    def translate_command(self, command, context=None):
        """
        Translate a high-level command to ROS 2 action call
        """
        # 1. Parse the command
        parsed = self.parse_command(command)
        
        # 2. Map to appropriate ROS 2 action
        action_info = self.map_to_action(parsed)
        
        # 3. Set up parameters based on context
        params = self.set_parameters(parsed, context)
        
        # 4. Execute the action
        return self.execute_action(action_info, params)
    
    def parse_command(self, command):
        # Parse natural language command
        # Extract action, objects, locations, etc.
        pass
    
    def map_to_action(self, parsed_command):
        # Map semantic action to ROS 2 action interface
        pass
    
    def set_parameters(self, parsed_command, context):
        # Set action parameters based on command and context
        pass
    
    def execute_action(self, action_info, parameters):
        # Execute ROS 2 action with given parameters
        pass
```

### Semantic Action Mapping

High-level actions need to be mapped to specific ROS 2 action interfaces:

```python
SEMANTIC_ACTION_MAPPING = {
    "navigate": {
        "action_server": "/navigate_to_pose",
        "action_type": "nav2_msgs.action.NavigateToPose",
        "parameter_mapping": {
            "location": "pose.pose.position",
            "orientation": "pose.pose.orientation"
        }
    },
    "pick_up": {
        "action_server": "/pick_and_place",
        "action_type": "manipulation_msgs.action.PickAndPlace",
        "parameter_mapping": {
            "object": "pick_object",
            "pose": "pick_pose"
        }
    },
    "grasp": {
        "action_server": "/gripper_command",
        "action_type": "control_msgs.action.GripperCommand",
        "parameter_mapping": {
            "width": "command.position",
            "effort": "command.max_effort"
        }
    },
    "move_arm": {
        "action_server": "/arm_controller/follow_joint_trajectory",
        "action_type": "control_msgs.action.FollowJointTrajectory",
        "parameter_mapping": {
            "joint_positions": "trajectory.joint_names",
            "joint_values": "trajectory.points.positions"
        }
    }
}
```

### Context-Aware Translation

The same command may require different action parameters based on context:

```python
class ContextManager:
    def __init__(self):
        self.current_state = {}
        self.environment_map = {}
        self.robot_capabilities = {}
    
    def get_contextual_parameters(self, command, environment_state):
        """
        Get action parameters based on current context
        """
        params = {}
        
        # Map locations to coordinates based on environment map
        if "location" in command:
            location_name = command["location"]
            params["pose"] = self.get_coordinates(location_name, environment_state)
        
        # Adjust parameters based on robot capabilities
        if "object" in command:
            object_info = self.get_object_info(command["object"], environment_state)
            params["approach_vector"] = self.calculate_approach_vector(object_info)
        
        # Consider current robot state
        if self.current_state.get("gripper", "open") == "closed":
            params["pre_grasp_position"] = self.calculate_pre_grasp()
        
        return params
```

## Implementing ROS 2 Actions

### Creating Custom Actions

To implement custom actions, we need to define action interface files (`.action`) and implement the server/client:

**Navigation action definition (NavigateToObject.action)**:
```
# Goal: Send an object name to navigate to
string object_name

---
# Result: Success or failure 
bool success
string message

---
# Feedback: Provide current progress
float32 progress
string status
geometry_msgs/Pose current_pose
```

**Action Server Implementation**:
```python
import rclpy
from rclpy.action import ActionServer
from rclpy.node import Node
from my_robot_msgs.action import NavigateToObject
from geometry_msgs.msg import Pose
import math

class NavigateToObjectActionServer(Node):
    def __init__(self):
        super().__init__('navigate_to_object_server')
        
        # Create action server
        self._action_server = ActionServer(
            self,
            NavigateToObject,
            'navigate_to_object',
            self.execute_callback
        )
        
        # Create navigation interface
        self.nav_client = self.create_client(NavigateToPose, '/navigate_to_pose')
        
    def execute_callback(self, goal_handle):
        self.get_logger().info(f'Receiving goal request to navigate to {goal_handle.request.object_name}')
        
        # Find object in environment
        object_pose = self.find_object_pose(goal_handle.request.object_name)
        if object_pose is None:
            goal_handle.abort()
            result = NavigateToObject.Result()
            result.success = False
            result.message = f'Object {goal_handle.request.object_name} not found'
            return result
        
        # Compute approach pose
        approach_pose = self.compute_approach_pose(object_pose)
        
        # Execute navigation
        self.get_logger().info('Executing navigation...')
        
        # Send navigation goal
        nav_goal = NavigateToPose.Goal()
        nav_goal.pose = approach_pose
        
        # Wait for navigation to complete with periodic feedback
        while not nav_client.wait_for_service(timeout_sec=1.0):
            self.get_logger().info('Navigation service not available, waiting...')
        
        future = nav_client.call_async(nav_goal)
        
        # Monitor progress and provide feedback
        while rclpy.ok():
            if future.done():
                break
            
            # Calculate progress based on current robot position vs target
            current_pose = self.get_current_robot_pose()
            distance = self.calculate_distance(current_pose, approach_pose)
            progress = max(0.0, min(1.0, 1.0 - distance / initial_distance))
            
            # Provide feedback
            feedback = NavigateToObject.Feedback()
            feedback.progress = progress
            feedback.status = "Navigating"
            feedback.current_pose = current_pose
            goal_handle.publish_feedback(feedback)
            
            time.sleep(0.1)  # Feedback update rate
        
        # Return result
        nav_result = future.result()
        result = NavigateToObject.Result()
        result.success = nav_result.success
        result.message = nav_result.message if nav_result.success else "Navigation failed"
        
        if result.success:
            goal_handle.succeed()
        else:
            goal_handle.abort()
        
        return result
```

**Action Client Implementation**:
```python
import rclpy
from rclpy.action import ActionClient
from rclpy.node import Node
from my_robot_msgs.action import NavigateToObject

class ActionClientNode(Node):
    def __init__(self):
        super().__init__('navigate_to_object_client')
        self._action_client = ActionClient(
            self,
            NavigateToObject,
            'navigate_to_object'
        )

    def send_goal(self, object_name):
        # Wait for action server
        self._action_client.wait_for_server()
        
        # Create goal
        goal_msg = NavigateToObject.Goal()
        goal_msg.object_name = object_name
        
        # Send goal and get future
        self._send_goal_future = self._action_client.send_goal_async(
            goal_msg,
            feedback_callback=self.feedback_callback
        )
        
        self._send_goal_future.add_done_callback(self.goal_response_callback)

    def goal_response_callback(self, future):
        goal_handle = future.result()
        if not goal_handle.accepted:
            self.get_logger().info('Goal rejected')
            return

        self.get_logger().info('Goal accepted')
        self._get_result_future = goal_handle.get_result_async()
        self._get_result_future.add_done_callback(self.get_result_callback)

    def get_result_callback(self, future):
        result = future.result().result
        self.get_logger().info(f'Result: {result.success}, {result.message}')

    def feedback_callback(self, feedback_msg):
        self.get_logger().info(
            f'Feedback: {feedback_msg.progress * 100:.1f}% - {feedback_msg.status}'
        )
```

## Translation Examples

### Navigation Translation

Converting natural language commands to navigation actions:

```python
class NavigationTranslator:
    def __init__(self, node):
        self.node = node
        self.action_client = ActionClient(
            node, NavigateToPose, '/navigate_to_pose'
        )
        self.location_mapping = self.load_location_map()
    
    def translate_navigation_command(self, command):
        """
        Translate commands like "Go to the kitchen" or 
        "Navigate to the red cup" to NavigateToPose action
        """
        # Parse location from command
        target_location = self.extract_location(command)
        
        if target_location in self.location_mapping:
            # Use predefined location
            target_pose = self.location_mapping[target_location]
        else:
            # Try to find object in environment
            target_pose = self.find_object_pose(target_location)
        
        if target_pose is None:
            return {"success": False, "message": f"Could not locate {target_location}"}
        
        # Create navigation goal
        goal_msg = NavigateToPose.Goal()
        goal_msg.pose = target_pose
        
        # Send action
        self.action_client.wait_for_server()
        future = self.action_client.send_goal_async(goal_msg)
        
        return {"success": True, "future": future}
    
    def extract_location(self, command):
        # Simple location extraction - in practice, use NLP
        import re
        location_patterns = [
            r"to the (.+)",
            r"to (.+)",
            r"go to (.+)",
            r"navigate to (.+)"
        ]
        
        for pattern in location_patterns:
            match = re.search(pattern, command.lower())
            if match:
                return match.group(1).strip()
        
        return command.strip()
```

### Manipulation Translation

Converting manipulation commands to appropriate action calls:

```python
class ManipulationTranslator:
    def __init__(self, node):
        self.node = node
        self.pick_action_client = ActionClient(
            node, PickAndPlace, '/pick_and_place'
        )
        self.gripper_client = ActionClient(
            node, GripperCommand, '/gripper_command'
        )
    
    def translate_manipulation_command(self, command, context=None):
        """
        Translate commands like "Pick up the red cup" or 
        "Grasp the object on the table"
        """
        # Parse manipulation action
        action_type, object_desc = self.parse_manipulation(command)
        
        if action_type == "grasp":
            return self.execute_grasp(object_desc, context)
        elif action_type == "place":
            return self.execute_place(object_desc, context)
        elif action_type == "pick":
            return self.execute_pick(object_desc, context)
        else:
            return {"success": False, "message": f"Unknown manipulation: {action_type}"}
    
    def execute_grasp(self, object_desc, context):
        # Find object in environment
        obj_info = self.perception_system.find_object(object_desc, context)
        if not obj_info:
            return {"success": False, "message": f"Could not find {object_desc}"}
        
        # Calculate approach and grasp poses
        approach_pose = self.calculate_approach_pose(obj_info)
        grasp_pose = obj_info["pose"]
        
        # Execute gripper action
        goal_msg = GripperCommand.Goal()
        goal_msg.command.position = 0.0  # Fully closed for grasp
        goal_msg.command.max_effort = 50.0
        
        self.gripper_client.wait_for_server()
        future = self.gripper_client.send_goal_async(goal_msg)
        
        return {"success": True, "future": future}
    
    def execute_pick(self, object_desc, context):
        # Use the integrated pick-and-place action
        obj_info = self.perception_system.find_object(object_desc, context)
        if not obj_info:
            return {"success": False, "message": f"Could not find {object_desc}"}
        
        goal_msg = PickAndPlace.Goal()
        goal_msg.object_name = object_desc
        goal_msg.object_pose = obj_info["pose"]
        goal_msg.approach_vector = obj_info["approach_vector"]
        
        self.pick_action_client.wait_for_server()
        future = self.pick_action_client.send_goal_async(goal_msg)
        
        return {"success": True, "future": future}
    
    def parse_manipulation(self, command):
        # Simple command parsing - in practice, use more sophisticated NLP
        if "grasp" in command.lower() or "grab" in command.lower():
            return "grasp", command.replace("grasp", "").replace("grab", "").strip()
        elif "place" in command.lower() or "put" in command.lower():
            return "place", command.replace("place", "").replace("put", "").strip()
        elif "pick" in command.lower():
            return "pick", command.replace("pick up", "").replace("pick", "").strip()
        else:
            return "unknown", command
```

## Integration with LLM Systems

### LLM to ROS 2 Pipeline

```python
class LLMToROS2Bridge:
    def __init__(self, llm_planner, ros_node):
        self.llm_planner = llm_planner
        self.ros_node = ros_node
        
        # Initialize translators for different action types
        self.nav_translator = NavigationTranslator(ros_node)
        self.manip_translator = ManipulationTranslator(ros_node)
        self.other_translators = {}
    
    def execute_natural_language_command(self, command, environment_context=None):
        """
        Execute a natural language command end-to-end
        """
        # 1. Use LLM to generate high-level plan
        plan = self.llm_planner.generate_plan(command, environment_context)
        
        # 2. Execute plan step by step
        results = []
        for step in plan["steps"]:
            result = self.execute_plan_step(step, environment_context)
            results.append(result)
            
            if not result["success"]:
                # Handle failure
                recovery_plan = self.llm_planner.generate_recovery_plan(
                    step, result["error"], environment_context
                )
                
                if recovery_plan:
                    recovery_result = self.execute_recovery_plan(
                        recovery_plan, environment_context
                    )
                    results.append(recovery_result)
                else:
                    break  # No recovery possible
        
        return results
    
    def execute_plan_step(self, step, context):
        """
        Execute a single step from the LLM-generated plan
        """
        action_type = step["action"]
        parameters = step["parameters"]
        
        if action_type in ["navigate", "move_to", "go_to"]:
            return self.nav_translator.translate_navigation_command(
                f"go to {parameters.get('location', parameters.get('target', 'unknown'))}"
            )
        elif action_type in ["pick", "grasp", "pickup", "grab"]:
            return self.manip_translator.translate_manipulation_command(
                f"pick {parameters.get('object', 'unknown')}", context
            )
        elif action_type == "place":
            return self.manip_translator.translate_manipulation_command(
                f"place {parameters.get('object', 'unknown')} at {parameters.get('location', 'default')}", context
            )
        else:
            # For other action types, implement appropriate translators
            return {"success": False, "message": f"Unknown action type: {action_type}"}
```

### Real-Time Adaptation

Handling dynamic changes during execution:

```python
class AdaptiveActionTranslator:
    def __init__(self, base_translator, perception_system):
        self.base_translator = base_translator
        self.perception = perception_system
        self.interrupt_handler = InterruptHandler()
    
    def execute_action_with_monitoring(self, action_spec, context):
        """
        Execute action while monitoring environment for changes
        """
        # Start action execution
        execution_handle = self.base_translator.start_execution(action_spec, context)
        
        # Monitor for environmental changes
        monitor_thread = threading.Thread(
            target=self._monitor_environment,
            args=(execution_handle, context)
        )
        monitor_thread.start()
        
        # Wait for completion or interruption
        result = execution_handle.wait_for_completion()
        
        # Stop monitoring
        self.interrupt_handler.stop_monitoring()
        monitor_thread.join()
        
        return result
    
    def _monitor_environment(self, execution_handle, context):
        """
        Monitor environment and interrupt execution if needed
        """
        while execution_handle.is_active():
            # Check for obstacles in navigation path
            if execution_handle.action_type == "navigate":
                obstacles = self.perception.get_path_obstacles(
                    execution_handle.current_path
                )
                
                if obstacles:
                    # Plan around obstacle
                    new_path = self.perception.plan_around_obstacles(
                        execution_handle.start_pose,
                        execution_handle.goal_pose,
                        obstacles
                    )
                    
                    if new_path:
                        execution_handle.update_path(new_path)
            
            # Check for object movement during manipulation
            elif execution_handle.action_type == "manipulation":
                target_obj = execution_handle.target_object
                current_pose = self.perception.get_object_pose(target_obj)
                expected_pose = execution_handle.expected_pose
                
                pose_diff = self._calculate_pose_difference(current_pose, expected_pose)
                
                if pose_diff > self.MOTION_THRESHOLD:
                    # Object moved significantly, replan
                    execution_handle.interrupt_and_replan(current_pose)
        
            time.sleep(0.1)  # Monitor frequency
```

## Advanced Translation Techniques

### Multi-Step Action Sequences

Some complex commands require sequences of actions:

```python
class SequentialActionExecutor:
    def __init__(self, node):
        self.node = node
        self.action_queue = []
        self.current_action = None
    
    def execute_complex_command(self, command, context):
        """
        Execute a complex command that requires multiple actions
        """
        # Break down complex command into sequence of actions
        action_sequence = self._decompose_command(command, context)
        
        # Execute sequence
        results = []
        for action_spec in action_sequence:
            result = self._execute_single_action(action_spec, context)
            results.append(result)
            
            if not result["success"]:
                # Handle failure based on command structure
                if self._is_critical_action(action_spec):
                    return {"success": False, "partial_results": results}
                else:
                    self.node.get_logger().warning(
                        f"Non-critical action failed: {result}"
                    )
        
        return {"success": True, "results": results}
    
    def _decompose_command(self, command, context):
        """
        Decompose a command into a sequence of ROS 2 actions
        """
        if "set the table" in command.lower():
            # Example: set the table requires multiple navigation and manipulation steps
            return [
                {"action": "navigate", "params": {"location": "cabinet"}},
                {"action": "grasp", "params": {"object": "plate1"}},
                {"action": "navigate", "params": {"location": "table"}},
                {"action": "place", "params": {"object": "plate1", "location": "table", "position": "seat1"}},
                
                {"action": "navigate", "params": {"location": "cabinet"}},
                {"action": "grasp", "params": {"object": "fork1"}},
                {"action": "navigate", "params": {"location": "table"}},
                {"action": "place", "params": {"object": "fork1", "location": "table", "position": "seat1"}},
                # ... continue for all necessary items
            ]
        elif "bring me water" in command.lower():
            return [
                {"action": "navigate", "params": {"location": "kitchen"}},
                {"action": "grasp", "params": {"object": "cup"}},
                {"action": "navigate", "params": {"location": "water_source"}},
                {"action": "fill", "params": {"object": "cup", "liquid": "water"}},
                {"action": "navigate", "params": {"location": "user"}},
                {"action": "place", "params": {"location": "user_table"}},
            ]
        
        # Default: treat as single action
        return [{"action": "default", "params": {"command": command}}]
```

### Parallel Action Execution

For efficiency, some actions can be executed in parallel:

```python
class ParallelActionExecutor:
    def __init__(self, node):
        self.node = node
        self.action_clients = {}
    
    def execute_parallel_actions(self, action_list, context):
        """
        Execute multiple actions in parallel where possible
        """
        # Group actions by resource requirements
        resource_groups = self._group_by_resources(action_list)
        
        # Execute each group in parallel
        all_results = {}
        for resource, actions in resource_groups.items():
            if len(actions) == 1:
                # Single action - execute normally
                result = self._execute_single_action(actions[0])
                all_results[actions[0]["id"]] = result
            else:
                # Multiple actions - execute in parallel
                thread_results = {}
                threads = []
                
                for action in actions:
                    thread = threading.Thread(
                        target=self._execute_and_store,
                        args=(action, thread_results, context)
                    )
                    threads.append(thread)
                    thread.start()
                
                # Wait for all threads to complete
                for thread in threads:
                    thread.join()
                
                all_results.update(thread_results)
        
        return all_results
    
    def _group_by_resources(self, action_list):
        """
        Group actions that can be executed in parallel vs. sequential
        """
        groups = {}
        
        for action in action_list:
            # Determine resource requirements
            resource_key = self._get_resource_requirements(action)
            
            if resource_key not in groups:
                groups[resource_key] = []
            
            # Check for conflicts with existing actions in group
            can_add = True
            for existing_action in groups[resource_key]:
                if self._has_resource_conflict(action, existing_action):
                    # Create new group for conflicting action
                    new_key = f"{resource_key}_conflict_{len(groups)}"
                    if new_key not in groups:
                        groups[new_key] = []
                    groups[new_key].append(action)
                    can_add = False
                    break
            
            if can_add:
                groups[resource_key].append(action)
        
        return groups
    
    def _get_resource_requirements(self, action):
        """
        Determine what resources an action requires
        """
        if action["action"] in ["navigate", "move_to"]:
            return "navigation"
        elif action["action"] in ["grasp", "place", "manipulate"]:
            return "manipulation_arm"
        elif action["action"] in ["open_gripper", "close_gripper"]:
            return "gripper"
        else:
            return "other"
```

## Error Handling and Recovery

### Action Failure Handling

```python
class ActionFailureHandler:
    def __init__(self, translator_system):
        self.translator = translator_system
        self.failure_patterns = {}
        self.recovery_strategies = {}
    
    def handle_action_failure(self, action_spec, error_message, context):
        """
        Handle failure of a ROS 2 action and attempt recovery
        """
        # Classify the failure type
        failure_type = self._classify_failure(error_message)
        
        # Look up recovery strategy
        recovery_strategy = self.recovery_strategies.get(
            f"{action_spec['action']}_{failure_type}", 
            self._default_recovery
        )
        
        # Execute recovery
        recovery_result = recovery_strategy(action_spec, error_message, context)
        
        return recovery_result
    
    def _classify_failure(self, error_message):
        """
        Classify the type of failure based on error message
        """
        error_lower = error_message.lower()
        
        if "collision" in error_lower or "obstacle" in error_lower:
            return "obstacle_detected"
        elif "timeout" in error_lower:
            return "timeout"
        elif "gripper" in error_lower and ("object" in error_lower or "grasp" in error_lower):
            return "grasp_failure"
        elif "navigation" in error_lower or "path" in error_lower:
            return "navigation_failure"
        else:
            return "unknown"
    
    def _default_recovery(self, action_spec, error_message, context):
        """
        Default recovery strategy
        """
        # Retry with modified parameters
        modified_spec = action_spec.copy()
        
        # For navigation failures, try alternative paths
        if action_spec["action"] in ["navigate", "move_to"]:
            alternative_goals = self._find_alternative_goals(
                action_spec["params"]["location"], 
                context
            )
            
            for alt_goal in alternative_goals:
                modified_spec["params"]["location"] = alt_goal
                try:
                    result = self.translator.execute_action(modified_spec, context)
                    if result["success"]:
                        return result
                except Exception:
                    continue
        
        return {"success": False, "error": error_message, "tried_recovery": True}
```

## Performance Optimization

### Caching and Prediction

```python
class OptimizedActionTranslator:
    def __init__(self, base_translator):
        self.base_translator = base_translator
        self.translation_cache = {}
        self.performance_predictor = PerformancePredictor()
    
    def translate_with_optimization(self, command, context):
        """
        Translate command with performance optimization
        """
        # Check cache first
        cache_key = self._generate_cache_key(command, context)
        if cache_key in self.translation_cache:
            cached_result = self.translation_cache[cache_key]
            
            # Verify cache is still valid
            if self._cache_valid_for_context(cached_result, context):
                return cached_result
        
        # Predict performance for different translation strategies
        strategies = self._generate_translation_strategies(command)
        best_strategy = self.performance_predictor.predict_best_strategy(
            strategies, context
        )
        
        # Execute best strategy
        result = self.base_translator.translate_with_strategy(
            command, context, best_strategy
        )
        
        # Cache result
        self.translation_cache[cache_key] = result
        
        return result
    
    def _generate_cache_key(self, command, context):
        """
        Generate a cache key for the command and context
        """
        import hashlib
        key_str = f"{command}_{str(sorted(context.items()))}"
        return hashlib.md5(key_str.encode()).hexdigest()
    
    def _generate_translation_strategies(self, command):
        """
        Generate multiple possible translation strategies for a command
        """
        strategies = []
        
        # Strategy 1: Direct translation
        strategies.append({
            "name": "direct",
            "steps": [command]
        })
        
        # Strategy 2: Decomposed translation
        strategies.append({
            "name": "decomposed",
            "steps": self._decompose_command(command)
        })
        
        # Strategy 3: Context-aware translation
        strategies.append({
            "name": "context_aware",
            "steps": self._decompose_with_context(command)
        })
        
        return strategies
```

## Integration Patterns

### Middleware Integration

```python
class MiddlewareIntegratedTranslator:
    def __init__(self, ros2_node, middleware_adapters):
        self.ros2_node = ros2_node
        self.adapters = middleware_adapters  # Other systems like perception, planning
    
    def translate_with_context(self, command):
        """
        Translate command using context from multiple middleware components
        """
        # Gather context from all integrated systems
        environment_context = self.adapters["perception"].get_environment_state()
        robot_state = self.adapters["robot_state"].get_current_state()
        map_info = self.adapters["map_manager"].get_map_info()
        
        context = {
            "environment": environment_context,
            "robot": robot_state,
            "map": map_info
        }
        
        # Translate command using comprehensive context
        return self._translate_command_with_context(command, context)
```

## Real-World Implementation Example

Here's a complete example of an action translation system:

```python
import rclpy
from rclpy.action import ActionClient, ActionServer
from rclpy.node import Node
from geometry_msgs.msg import Pose, Point, Quaternion
from std_msgs.msg import String
import json
import threading
import time

class ActionTranslationSystem(Node):
    def __init__(self):
        super().__init__('action_translation_system')
        
        # ROS 2 action clients
        self.nav_client = ActionClient(self, NavigateToPose, '/navigate_to_pose')
        self.manip_client = ActionClient(self, PickAndPlace, '/pick_and_place')
        
        # Publisher for high-level commands
        self.command_sub = self.create_subscription(
            String,
            '/high_level_command',
            self.command_callback,
            10
        )
        
        # Publisher for execution status
        self.status_pub = self.create_publisher(String, '/action_status', 10)
        
        # Translation cache and context
        self.translation_cache = {}
        self.context = {}
        
    def command_callback(self, msg):
        """
        Receive high-level command and translate to ROS 2 actions
        """
        try:
            # Parse command from JSON string
            command_data = json.loads(msg.data)
            command_type = command_data["type"]
            command_params = command_data["params"]
            
            self.get_logger().info(f"Received command: {command_type}")
            
            # Translate and execute
            result = self.translate_and_execute(command_type, command_params)
            
            # Publish status
            status_msg = String()
            status_msg.data = json.dumps({
                "command": command_type,
                "success": result["success"],
                "message": result.get("message", "")
            })
            self.status_pub.publish(status_msg)
            
        except Exception as e:
            self.get_logger().error(f"Error processing command: {e}")
    
    def translate_and_execute(self, command_type, params):
        """
        Main translation and execution method
        """
        if command_type == "navigate_to":
            return self.execute_navigation(params)
        elif command_type == "pick_object":
            return self.execute_manipulation(params)
        elif command_type == "place_object":
            return self.execute_placement(params)
        else:
            return {"success": False, "message": f"Unknown command type: {command_type}"}
    
    def execute_navigation(self, params):
        """
        Execute navigation command
        """
        # Resolve location name to coordinates
        location_name = params.get("location", "")
        target_pose = self.resolve_location_to_pose(location_name)
        
        if target_pose is None:
            return {"success": False, "message": f"Unknown location: {location_name}"}
        
        # Wait for navigation server
        if not self.nav_client.wait_for_server(timeout_sec=5.0):
            return {"success": False, "message": "Navigation server not available"}
        
        # Create and send goal
        goal_msg = NavigateToPose.Goal()
        goal_msg.pose = target_pose
        
        future = self.nav_client.send_goal_async(
            goal_msg,
            feedback_callback=self.navigation_feedback_callback
        )
        
        # Wait for result (with timeout)
        rclpy.spin_until_future_complete(self, future, timeout_sec=60.0)
        
        if future.done():
            result = future.result()
            if result.success:
                return {"success": True, "message": "Navigation completed successfully"}
            else:
                return {"success": False, "message": "Navigation failed"}
        else:
            return {"success": False, "message": "Navigation timed out"}
    
    def navigation_feedback_callback(self, feedback_msg):
        """
        Handle navigation feedback
        """
        self.get_logger().info(f"Navigation progress: {feedback_msg.feedback.distance_remaining:.2f}m remaining")
    
    def execute_manipulation(self, params):
        """
        Execute manipulation command
        """
        # For this example, we'll simulate the manipulation
        object_name = params.get("object", "")
        self.get_logger().info(f"Attempting to manipulate object: {object_name}")
        
        # Simulate manipulation action (would connect to real manipulator in practice)
        time.sleep(2.0)  # Simulate action time
        
        # Return success
        return {"success": True, "message": f"Manipulation of {object_name} completed"}
    
    def resolve_location_to_pose(self, location_name):
        """
        Resolve a location name to a Pose
        """
        # In a real system, this would look up in a map or use perception
        location_map = {
            "kitchen": Pose(position=Point(x=1.0, y=2.0, z=0.0), 
                           orientation=Quaternion(w=1.0, x=0.0, y=0.0, z=0.0)),
            "living_room": Pose(position=Point(x=3.0, y=0.0, z=0.0), 
                              orientation=Quaternion(w=1.0, x=0.0, y=0.0, z=0.0)),
            "bedroom": Pose(position=Point(x=5.0, y=3.0, z=0.0), 
                           orientation=Quaternion(w=1.0, x=0.0, y=0.0, z=0.0)),
        }
        
        return location_map.get(location_name.lower())

def main(args=None):
    rclpy.init(args=args)
    
    action_translator = ActionTranslationSystem()
    
    try:
        rclpy.spin(action_translator)
    except KeyboardInterrupt:
        pass
    finally:
        action_translator.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()
```

## Testing and Validation

### Unit Testing Action Translation

```python
import unittest
from unittest.mock import Mock, MagicMock

class TestActionTranslation(unittest.TestCase):
    def setUp(self):
        self.mock_node = Mock()
        self.translator = ActionTranslator()
        self.translator.nav_client = Mock()
        self.translator.manip_client = Mock()
    
    def test_navigation_translation(self):
        """Test translation of navigation commands"""
        command = "go to the kitchen"
        result = self.translator.translate_navigation_command(command)
        
        # Verify the action client was called with correct parameters
        self.assertTrue(self.translator.nav_client.send_goal_async.called)
        self.assertTrue(result["success"])
    
    def test_manipulation_translation(self):
        """Test translation of manipulation commands"""
        command = "pick up the red cup"
        context = {"objects": [{"name": "red cup", "pose": Mock()}]}
        
        result = self.translator.translate_manipulation_command(command, context)
        
        # Verify appropriate action was selected
        self.assertTrue(result["success"])
    
    def test_command_parsing(self):
        """Test parsing of various command formats"""
        test_cases = [
            ("go to the kitchen", "kitchen"),
            ("navigate to bedroom", "bedroom"), 
            ("move to living room", "living room"),
        ]
        
        for command, expected_location in test_cases:
            translator = NavigationTranslator(Mock())
            location = translator.extract_location(command)
            self.assertEqual(location, expected_location)
```

## Summary

ROS 2 action translation is a critical component in Physical AI systems, bridging the gap between high-level reasoning and low-level robot control. Effective action translation requires:

1. Understanding of ROS 2 action architecture and communication patterns
2. Semantic mapping between high-level commands and specific ROS 2 interfaces
3. Context-aware parameter setting based on environment and robot state
4. Robust error handling and recovery mechanisms
5. Performance optimization through caching and prediction
6. Integration with perception and planning systems

The translation layer enables flexible, natural interaction with robots while maintaining the modularity and reliability advantages of the ROS 2 framework. As Physical AI systems become more sophisticated, the action translation system will continue to evolve, incorporating advances in natural language processing, machine learning, and robotic control.

The next section will explore how to integrate multiple modalities in the vision-language-action pipeline, creating more robust and capable Physical AI systems.

## References

ROS.org. (2023). ROS 2 Actions Documentation. Retrieved from https://docs.ros.org/en/rolling/Tutorials/Actions.html

Quigley, M., Conley, K., Gerkey, B., Faust, J., Foote, T., Leibs, J., ... & Wheeler, D. (2009). ROS: an open-source Robot Operating System. ICRA Workshop on Open Source Software.

Macenski, S., et al. (2022). Navigation: The ROS 2 Navigation Stack. Journal of Open Source Software.

ROS.org. (2023). ROS 2 Design - Quality of Service. Retrieved from https://design.ros2.org/articles/qos.html

## Exercises

1. Implement an action translation system that can handle a sequence of navigation and manipulation commands. Test it with a simulated robot, verifying that commands like "Go to the kitchen, pick up the cup, then go to the living room and place the cup on the table" are correctly executed.

2. Design a context-aware action translation system that uses real-time perception data to modify action parameters. For example, if a planned navigation location becomes blocked, the system should find an alternative route.

3. Create a performance-optimized action translation system with caching and prediction capabilities. Test how caching affects response time for repeated commands and how prediction improves execution efficiency.