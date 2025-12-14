---
sidebar_position: 2
---

# LLM-Based Planning Systems

## Introduction to LLM-Based Planning in Physical AI

Large Language Models (LLMs) represent a revolutionary advancement in artificial intelligence, capable of understanding and generating human language with unprecedented sophistication. When integrated into Physical AI systems, LLMs can serve as high-level reasoning and planning engines, translating natural language goals into executable robotic actions.

Unlike traditional planning systems that require formal domain specifications and manual state-action definitions, LLM-based planning systems leverage the vast knowledge encoded in these models to understand complex tasks, reason about physical constraints, and generate appropriate action sequences. This capability enables more intuitive human-robot interaction and simplifies the development of complex robotic behaviors.

## Fundamentals of LLM Planning

### The Planning Problem in Robotics

Traditional robotic planning involves:
- **State Space**: Representation of possible robot and environment configurations
- **Action Space**: Set of possible robot actions with defined preconditions and effects
- **Goal Specification**: Desired end-state or condition
- **Search Algorithm**: Method for finding a sequence of actions from start to goal

LLM-based planning transforms this by using language as the medium for representing states, actions, and goals, allowing for:
- Natural language goal specification
- Intuitive task decomposition
- Common-sense reasoning about physical constraints
- Flexible action selection based on context

### LLM Capabilities for Planning

**Knowledge Integration**:
- Access to vast amounts of common-sense knowledge
- Physical reasoning about objects and their interactions
- Understanding of social and cultural norms

**Reasoning and Inference**:
- Multi-step logical reasoning
- Analogical reasoning based on similar situations
- Handling of ambiguity and uncertainty

**Natural Language Processing**:
- Understanding of complex, ambiguous instructions
- Generation of natural language explanations
- Flexible interaction with human users

## LLM Planning Architecture

### High-Level Planning with LLMs

**Plan Structure**:
```
Goal: "Set the table for dinner"
├── Decompose into subtasks:
│   ├── Identify required items ("plates", "forks", "knives", "glasses")
│   ├── Locate items in environment
│   ├── Navigate to item locations
│   ├── Grasp and transport items
│   └── Place items at appropriate positions on table
└── Execute in sequence with appropriate error handling
```

### Integration Architecture

**LLM Planning Module**:
```python
class LLMPlanningModule:
    def __init__(self, model_name="gpt-3.5-turbo", api_key=None):
        self.model_name = model_name
        if api_key:
            openai.api_key = api_key
        self.context_history = []
    
    def generate_plan(self, goal, environment_state, robot_capabilities):
        """
        Generate a high-level plan for achieving the goal
        """
        prompt = self._construct_planning_prompt(
            goal, environment_state, robot_capabilities
        )
        
        response = openai.ChatCompletion.create(
            model=self.model_name,
            messages=[
                {"role": "system", "content": self._get_system_prompt()},
                {"role": "user", "content": prompt}
            ],
            temperature=0.1,
            max_tokens=1000
        )
        
        plan = self._parse_plan_from_response(response.choices[0].message.content)
        return plan
    
    def _construct_planning_prompt(self, goal, environment_state, robot_capabilities):
        prompt = f"""
        Goal: {goal}
        Environment State: {environment_state}
        Robot Capabilities: {robot_capabilities}
        
        Generate a detailed step-by-step plan to achieve the goal.
        Each step should be specific and executable by a robot.
        Return the plan in JSON format with the following structure:
        {{
            "plan": [
                {{
                    "step_number": 1,
                    "description": "Step description",
                    "action": "action_type",
                    "parameters": {{"param1": "value1", "param2": "value2"}},
                    "preconditions": ["condition1", "condition2"],
                    "expected_outcomes": ["outcome1", "outcome2"]
                }}
            ]
        }}
        """
        return prompt
    
    def _get_system_prompt(self):
        return """
        You are an expert robotic task planner. Your role is to decompose high-level goals 
        into detailed, executable steps for a robot. Consider the physical constraints 
        of the environment and the capabilities of the robot. Be specific about object 
        locations, manipulation actions, and navigation requirements. Return plans in JSON format.
        """
    
    def _parse_plan_from_response(self, response_text):
        # Parse JSON from response
        import json
        import re
        
        # Extract JSON from response if wrapped in markdown
        json_match = re.search(r'```json\n(.*?)\n```', response_text, re.DOTALL)
        if json_match:
            json_text = json_match.group(1)
        else:
            # Try to extract JSON directly
            start = response_text.find('{')
            end = response_text.rfind('}') + 1
            json_text = response_text[start:end]
        
        try:
            return json.loads(json_text)
        except json.JSONDecodeError:
            print(f"Error parsing JSON: {json_text}")
            return {"plan": []}
```

### Context and Memory Management

**Short-Term Context**:
- Current task and subtasks
- Robot state and environment observations
- Execution status and history

**Long-Term Memory**:
- Previously executed plans and outcomes
- Learned patterns and optimizations
- Environment-specific knowledge

## Planning Strategies and Techniques

### Hierarchical Task Decomposition

LLMs excel at breaking down complex goals into manageable subtasks:

```python
def hierarchical_decomposition(llm_planner, goal, max_depth=3):
    """
    Decompose a goal hierarchically using LLM reasoning
    """
    # First, get high-level plan
    high_level_plan = llm_planner.generate_plan(goal, {}, {})
    
    # For each step, further decompose if needed
    for step in high_level_plan["plan"]:
        if step["action"] == "complex_task":
            # Decompose the complex task further
            subtasks = llm_planner.generate_plan(
                step["description"], 
                {}, 
                {}
            )
            # Replace the complex task with its subtasks
            step["subtasks"] = subtasks["plan"]
    
    return high_level_plan
```

### Commonsense Physical Reasoning

LLMs incorporate commonsense knowledge about the physical world:

**Object Affordances**:
- Understanding what can be done with objects
- Grasping possibilities based on object shape and size
- Tool use and manipulation strategies

**Spatial Reasoning**:
- Understanding relative positions of objects
- Navigation to object locations
- Avoiding collisions and obstacles

**Causal Reasoning**:
- Predicting effects of actions
- Understanding tool-use relationships
- Planning for long-term consequences

### Handling Uncertainty

**Plan Robustness**:
```python
def generate_robust_plan(llm_planner, goal, uncertainty_factors):
    """
    Generate a plan with contingency options for uncertainty
    """
    prompt = f"""
    Goal: {goal}
    Uncertainty Factors: {uncertainty_factors}
    
    Generate a step-by-step plan that includes:
    1. Main execution path
    2. Contingency plans for likely failures
    3. Recovery strategies
    
    Format in JSON with fields:
    - "main_plan": primary sequence of actions
    - "contingency_plans": {{"failure_condition": [alternative_steps]}}
    - "recovery_strategies": [recovery_options]
    """
    
    response = openai.ChatCompletion.create(
        model="gpt-3.5-turbo",
        messages=[
            {"role": "system", "content": "Generate robust plans with contingencies"},
            {"role": "user", "content": prompt}
        ]
    )
    
    return json.loads(response.choices[0].message.content)
```

## LLM-Enhanced Classical Planning

### Integration with Classical Planners

LLMs can enhance traditional planning approaches by providing high-level guidance and domain knowledge:

```python
class HybridPlanningSystem:
    def __init__(self, llm_planner, classical_planner):
        self.llm_planner = llm_planner
        self.classical_planner = classical_planner
    
    def generate_hybrid_plan(self, goal, environment):
        # Use LLM for high-level task decomposition
        high_level_plan = self.llm_planner.generate_plan(goal, environment, {})
        
        # Convert LLM plan into classical planning problem
        classical_plan = []
        for step in high_level_plan["plan"]:
            if step["action"] in ["navigate", "grasp", "place"]:
                # Use classical planner for these specific tasks
                sub_plan = self.classical_planner.plan(
                    step["action"], step["parameters"], environment
                )
                classical_plan.extend(sub_plan)
            else:
                # Keep LLM-generated steps as high-level commands
                classical_plan.append(step)
        
        return classical_plan
```

### Knowledge-Guided Search

**Informed Search Heuristics**:
Use LLM knowledge to guide classical search algorithms:

```python
def knowledge_guided_heuristic(llm_client, state, goal):
    """
    Use LLM to provide heuristic estimates for classical planners
    """
    prompt = f"""
    Current State: {state}
    Goal: {goal}
    
    Estimate how close the current state is to the goal on a scale of 0 to 10,
    where 0 is completely different and 10 is achieved.
    
    Also list the most critical differences between current state and goal.
    """
    
    response = llm_client.completion(prompt)
    # Parse the response to extract heuristic value
    # This would need to be processed to extract a numerical value
    return heuristic_value
```

## LLM-Specific Planning Techniques

### Chain-of-Thought Planning

Use LLM's reasoning capabilities to explicitly plan through thought processes:

```python
def chain_of_thought_planning(llm_client, goal, environment):
    """
    Use chain-of-thought reasoning for complex planning
    """
    prompt = f"""
    Goal: {goal}
    Environment: {environment}
    
    Let's think step by step about how to achieve this goal:
    
    1) What is the current state?
    2) What is the desired end state?
    3) What are the intermediate states needed?
    4) What physical actions are required for each transition?
    5) What constraints must be considered?
    6) How can potential obstacles be overcome?
    
    Finally, provide a specific plan in JSON format.
    """
    
    response = llm_client.completion(prompt)
    
    # The LLM will work through the reasoning step by step before providing the plan
    return extract_plan_from_thought_process(response)
```

### Few-Shot Planning Examples

Provide examples to guide the LLM's planning behavior:

```python
def few_shot_planning(llm_client, goal, examples=None):
    """
    Use few-shot learning with examples for planning
    """
    if examples is None:
        examples = [
            {
                "goal": "Bring me a cup of water",
                "plan": [
                    {"action": "navigate", "params": {"location": "kitchen"}},
                    {"action": "grasp", "params": {"object": "cup"}},
                    {"action": "navigate", "params": {"location": "water_source"}},
                    {"action": "fill", "params": {"object": "cup", "liquid": "water"}},
                    {"action": "navigate", "params": {"location": "user"}},
                    {"action": "place", "params": {"location": "table"}}
                ]
            },
            {
                "goal": "Set the table for two people",
                "plan": [
                    {"action": "navigate", "params": {"location": "cabinet"}},
                    {"action": "grasp", "params": {"object": "plate"}},
                    {"action": "navigate", "params": {"location": "table"}},
                    {"action": "place", "params": {"location": "table", "position": "seat1"}},
                    # ... more steps
                ]
            }
        ]
    
    example_str = "\n\n".join([
        f"Goal: {ex['goal']}\nPlan: {json.dumps(ex['plan'])}" 
        for ex in examples
    ])
    
    prompt = f"""
    Here are examples of translating goals into robot action plans:
    
    {example_str}
    
    Now, for the following goal, provide a similar plan:
    
    Goal: {goal}
    Plan: (in same JSON format as examples)
    """
    
    response = llm_client.completion(prompt)
    return json.loads(response)
```

## Practical Implementation Considerations

### Performance Optimization

**Caching and Retrieval**:
```python
class CachedLLMPlanner:
    def __init__(self, base_planner, cache_size=100):
        self.base_planner = base_planner
        self.plan_cache = {}
        self.max_cache_size = cache_size
    
    def get_plan(self, goal):
        if goal in self.plan_cache:
            return self.plan_cache[goal]
        
        plan = self.base_planner.generate_plan(goal)
        
        # Implement LRU if cache is full
        if len(self.plan_cache) >= self.max_cache_size:
            # Remove oldest entry
            oldest_key = next(iter(self.plan_cache))
            del self.plan_cache[oldest_key]
        
        self.plan_cache[goal] = plan
        return plan
```

### Safety and Validation

**Plan Verification**:
```python
def verify_plan_with_llm(llm_client, proposed_plan, safety_constraints):
    """
    Use LLM to verify if a plan is safe and feasible
    """
    prompt = f"""
    Proposed Plan: {proposed_plan}
    Safety Constraints: {safety_constraints}
    
    Review this plan and identify:
    1. Any safety violations
    2. Logical inconsistencies
    3. Missing steps or conditions
    4. Potential failure points
    
    Respond with a safety assessment and any required modifications.
    """
    
    response = llm_client.completion(prompt)
    return response
```

### Error Handling and Recovery

**Failure Detection and Recovery**:
```python
class ResilientLLMPlanner:
    def __init__(self, base_planner):
        self.base_planner = base_planner
        self.failure_history = []
    
    def adapt_plan_on_failure(self, original_goal, executed_steps, failure_description):
        """
        Generate a new plan considering the failure
        """
        context = {
            "original_goal": original_goal,
            "executed_steps": executed_steps,
            "failure_description": failure_description,
            "past_failures": self.failure_history[-10:]  # Last 10 failures
        }
        
        prompt = f"""
        Context: {context}
        
        The robot failed to execute part of its plan. Generate a new plan that:
        1. Addresses the same original goal
        2. Avoids the failure that occurred
        3. Incorporates lessons from past failures when similar
        4. Includes additional checks to prevent similar failures
        
        Return the revised plan in JSON format.
        """
        
        new_plan = self.base_planner.generate_plan_from_prompt(prompt)
        return new_plan
```

## Integration with Physical AI Systems

### Perception Integration

LLM planning benefits from rich perception data:

```python
class PerceptionAwarePlanner:
    def __init__(self, llm_client, perception_system):
        self.llm_client = llm_client
        self.perception = perception_system
    
    def generate_contextual_plan(self, goal):
        # Get current environmental state from perception
        env_state = self.perception.get_environmental_state()
        
        # Include object detection results
        object_map = self.perception.get_object_map()
        
        # Get navigable areas
        navigation_map = self.perception.get_navigation_map()
        
        context = {
            "environmental_state": env_state,
            "object_locations": object_map,
            "navigable_areas": navigation_map
        }
        
        prompt = f"""
        Goal: {goal}
        Current Context: {context}
        
        Generate a plan that accounts for the current environment state,
        including specific object locations and navigable areas.
        """
        
        return self.llm_client.completion(prompt)
```

### Action Execution Interface

Bridge between high-level LLM plans and low-level robot actions:

```python
class PlanExecutor:
    def __init__(self, robot_interface):
        self.robot = robot_interface
    
    def execute_plan_step(self, step):
        """
        Convert high-level LLM plan step to robot action
        """
        action_type = step["action"]
        params = step["parameters"]
        
        if action_type == "navigate":
            return self.robot.navigate_to(params["location"])
        elif action_type == "grasp":
            return self.robot.grasp_object(params["object"])
        elif action_type == "place":
            return self.robot.place_object(params["object"], params["location"])
        elif action_type == "pick_up":
            return self.robot.pick_up(params["object"])
        elif action_type == "release":
            return self.robot.release()
        elif action_type == "look_at":
            return self.robot.look_at(params["location"])
        else:
            # For non-specific actions, use a general interface
            return self.robot.execute_action(action_type, params)
    
    def monitor_execution(self, plan):
        """
        Monitor plan execution and handle deviations
        """
        for i, step in enumerate(plan["plan"]):
            try:
                result = self.execute_plan_step(step)
                if not result:
                    # Handle failure
                    return self.handle_execution_failure(plan, i, step)
            except Exception as e:
                return self.handle_execution_exception(plan, i, step, str(e))
        
        return "Plan executed successfully"
```

## Evaluation and Benchmarking

### Planning Quality Metrics

**Completeness**:
- Whether the plan achieves the stated goal
- Coverage of all required subtasks
- Proper handling of preconditions

**Correctness**:
- Logical consistency of the plan
- Physical feasibility of actions
- Proper causal relationships

**Optimality**:
- Efficiency in terms of steps required
- Resource utilization
- Time to complete the task

### Human Evaluation

**Intuitiveness**:
- Whether the plan makes intuitive sense to humans
- Alignment with human expectations
- Naturalness of the approach

**Safety Assessment**:
- Identification of potential safety risks
- Adherence to safety constraints
- Inclusion of safety checks

## Challenges and Limitations

### Knowledge Limitations

**Training Data Constraints**:
- LLMs may lack specific domain knowledge
- Limited exposure to specialized robot behaviors
- Potential for hallucinating capabilities

**Physical Reasoning**:
- May not accurately model physics constraints
- Can suggest impossible or unsafe actions
- Requires validation against real physics

### Computational and Practical Issues

**Latency**:
- API call delays for cloud-based LLMs
- Processing time for complex planning queries
- Real-time constraints in robotics

**Cost**:
- Per-token costs for API-based LLMs
- Continuous usage in operational systems
- Need for cost-effective deployment strategies

### Safety and Reliability

**Verification Challenges**:
- Difficulty in validating complex LLM outputs
- Potential for unsafe action sequences
- Need for systematic safety checks

**Consistency**:
- Variability in LLM outputs
- Ensuring consistent behavior across runs
- Managing temperature and randomness settings

## Advanced Topics

### Multi-Agent Planning

Using LLMs for coordinating multiple robots:

```python
def multi_agent_plan(llm_client, joint_goal, agents_info):
    """
    Generate coordinated plan for multiple agents
    """
    prompt = f"""
    Joint Goal: {joint_goal}
    Agents: {agents_info}
    
    Generate a coordinated plan where:
    1. Each agent has specific responsibilities
    2. Agent actions are properly synchronized
    3. Resource conflicts are avoided
    4. Communication requirements are explicit
    
    Return plan in format: {{"agent1": [actions], "agent2": [actions], "coordination": [sync_points]}}
    """
    
    response = llm_client.completion(prompt)
    return json.loads(response)
```

### Learning from Execution

**Plan Improvement**:
```python
def learn_from_execution(llm_client, executed_plan, outcomes):
    """
    Use execution results to improve future planning
    """
    prompt = f"""
    Executed Plan: {executed_plan}
    Outcomes: {outcomes}
    
    Based on this execution experience, how should future plans be modified?
    1. What worked well?
    2. What failed and why?
    3. How should similar situations be handled differently?
    4. What additional steps might be needed?
    """
    
    insights = llm_client.completion(prompt)
    return insights
```

## Future Directions

### Specialized LLMs for Robotics

**Domain-Adapted Models**:
- Fine-tuning LLMs on robotics-specific data
- Integrating physics simulators with LLMs
- Specialized architectures for planning tasks

**On-Device LLMs**:
- Distilled models for edge deployment
- Quantized models for resource-constrained robots
- Efficient architectures for real-time planning

### Hybrid Reasoning Systems

**Symbolic-Neural Integration**:
- Combining LLM reasoning with symbolic planners
- Neuro-symbolic approaches to planning
- Hybrid architectures leveraging both approaches

**Causal Models**:
- Integration with causal reasoning systems
- Physics-informed LLM planning
- Counterfactual reasoning for planning

## Implementation Example: Complete LLM Planning System

```python
import openai
import json
import time
from typing import Dict, List, Any

class RoboticLLMPlanningSystem:
    def __init__(self, api_key: str, model: str = "gpt-3.5-turbo"):
        openai.api_key = api_key
        self.model = model
        self.conversation_history = []
        self.action_executor = None
        
    def set_action_executor(self, executor):
        self.action_executor = executor
    
    def plan_task(self, goal: str, environment_state: Dict[str, Any]) -> Dict[str, Any]:
        """Generate a complete plan for the given goal and environment."""
        
        system_prompt = """
        You are an expert robotic task planner. Your role is to decompose complex goals 
        into specific, executable robot actions. Consider:
        1. Physical constraints of the environment
        2. Robot capabilities (navigation, manipulation, perception)
        3. Safety requirements
        4. Efficiency of the plan
        5. Error handling and recovery
        
        Respond with a JSON-formatted plan containing:
        - 'steps': List of action steps with parameters
        - 'preconditions': Conditions required before execution
        - 'expected_outcomes': What should happen after each step
        - 'safety_checks': Safety validations for each step
        """
        
        user_prompt = f"""
        Goal: {goal}
        Environment State: {json.dumps(environment_state, indent=2)}
        
        Generate a detailed step-by-step plan that can be executed by a robot.
        """
        
        response = openai.ChatCompletion.create(
            model=self.model,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ],
            temperature=0.1,
            max_tokens=1500
        )
        
        # Parse the response
        raw_response = response.choices[0].message.content
        
        # Extract JSON from response (handle potential markdown wrapping)
        import re
        json_match = re.search(r'```json\n(.*?)\n```', raw_response, re.DOTALL)
        if json_match:
            json_str = json_match.group(1)
        else:
            # Try to find JSON in response
            start_idx = raw_response.find('{')
            end_idx = raw_response.rfind('}') + 1
            json_str = raw_response[start_idx:end_idx]
        
        try:
            plan = json.loads(json_str)
            return plan
        except json.JSONDecodeError:
            print("Failed to parse LLM response as JSON")
            print("Raw response:", raw_response)
            return {"steps": [], "error": "Failed to parse plan"}
    
    def execute_plan(self, plan: Dict[str, Any]) -> Dict[str, Any]:
        """Execute the generated plan step by step."""
        if not self.action_executor:
            return {"success": False, "error": "No action executor configured"}
        
        results = []
        success = True
        
        for i, step in enumerate(plan.get("steps", [])):
            print(f"Executing step {i+1}: {step.get('action', 'unknown')}")
            
            try:
                # Check preconditions
                if "preconditions" in step:
                    if not self._check_preconditions(step["preconditions"]):
                        return {"success": False, "error": f"Preconditions failed for step {i+1}"}
                
                # Execute the action
                action_result = self.action_executor.execute_step(step)
                results.append({
                    "step": i+1,
                    "action": step.get("action"),
                    "success": action_result.get("success", False),
                    "details": action_result
                })
                
                # Check if step was successful
                if not action_result.get("success", False):
                    success = False
                    break
                    
            except Exception as e:
                results.append({
                    "step": i+1,
                    "action": step.get("action"),
                    "success": False,
                    "error": str(e)
                })
                success = False
                break
        
        return {
            "success": success,
            "execution_log": results,
            "final_state": self.action_executor.get_robot_state()
        }
    
    def _check_preconditions(self, preconditions: List[str]) -> bool:
        """Check if preconditions are met before executing a step."""
        # This would interface with the robot's perception system
        # For now, we'll implement a basic check
        return True  # Placeholder implementation
    
    def refine_plan(self, original_plan: Dict[str, Any], failure_info: Dict[str, Any]) -> Dict[str, Any]:
        """Refine a plan based on execution failures."""
        
        system_prompt = """
        You are an expert robotic task planner specializing in plan refinement. 
        Based on execution failures, suggest how to improve the plan.
        Consider alternative approaches and recovery strategies.
        """
        
        user_prompt = f"""
        Original Plan: {json.dumps(original_plan, indent=2)}
        Failure Information: {json.dumps(failure_info, indent=2)}
        
        Suggest a refined plan that addresses the failures while achieving the same goal.
        """
        
        response = openai.ChatCompletion.create(
            model=self.model,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ],
            temperature=0.3,
            max_tokens=1000
        )
        
        # Parse the refined plan response
        raw_response = response.choices[0].message.content
        
        import re
        json_match = re.search(r'```json\n(.*?)\n```', raw_response, re.DOTALL)
        if json_match:
            json_str = json_match.group(1)
        else:
            start_idx = raw_response.find('{')
            end_idx = raw_response.rfind('}') + 1
            json_str = raw_response[start_idx:end_idx]
        
        try:
            return json.loads(json_str)
        except json.JSONDecodeError:
            print("Failed to parse refined plan")
            return original_plan  # Return original if refinement failed

# Example usage
if __name__ == "__main__":
    # Initialize the planning system
    planner = RoboticLLMPlanningSystem(api_key="your-api-key")
    
    # Example environment state (would come from perception system)
    env_state = {
        "robot_location": [0, 0, 0],
        "objects": [
            {"name": "apple", "type": "fruit", "location": [1, 1, 0], "pose": {}},
            {"name": "table", "type": "furniture", "location": [2, 2, 0], "pose": {}}
        ],
        "workspace_limits": {"x": [-5, 5], "y": [-5, 5], "z": [0, 2]}
    }
    
    # Plan a task
    goal = "Pick up the apple and place it on the table"
    plan = planner.plan_task(goal, env_state)
    
    print("Generated Plan:")
    print(json.dumps(plan, indent=2))
```

## Summary

LLM-based planning systems represent a powerful approach to high-level robot control, enabling natural language interaction and sophisticated reasoning capabilities. These systems can generate complex plans by leveraging the vast knowledge encoded in LLMs, handling ambiguity in natural language goals, and reasoning about physical constraints.

However, successful deployment requires careful integration with perception and control systems, validation of generated plans, and attention to safety considerations. The field continues to evolve with new techniques for improving the reliability and efficiency of LLM-driven robotic planning.

The next section will explore how to translate high-level plans generated by LLMs into specific ROS 2 actions that can be executed by robotic systems.

## References

Brown, T., Mann, B., Ryder, N., Subbiah, M., Kaplan, J. D., Dhariwal, P., ... & Amodei, D. (2020). Language models are few-shot learners. Advances in neural information processing systems.

Ahn, M., Brohan, A., Chebotar, Y., Finn, C., Fu, K., Ibarz, J., ... & Lee, K. H. (2022). A collaborative framework for language-guided robotic manipulation. arXiv preprint arXiv:2209.01197.

Huang, W., Abbeel, P., Pathak, D., & Mordatch, I. (2022). Language models as zero-shot planners: Extracting actionable knowledge for embodied agents. International Conference on Machine Learning.

Yao, S., Zhao, H., Yu, Y., Cao, D., Feng, J., & Sha, F. (2023). ReAct: Synergizing reasoning and acting in language models. International Conference on Learning Representations.

## Exercises

1. Implement an LLM-based planning system that can handle a simple household task (e.g., making coffee). Integrate it with a simulated robot and evaluate its effectiveness in generating executable plans.

2. Design a system that combines LLM planning with a classical planner (like PDDL-based planners) for a complex manipulation task. Compare the performance of the hybrid approach with pure LLM and pure classical planning.

3. Develop a plan refinement mechanism that learns from execution failures and adjusts future planning strategies. Test this system in a simulated environment with various failure modes.