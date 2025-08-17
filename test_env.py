"""
Test script for the Snake RL environment.
"""

import numpy as np
from src.snake_env import SnakeEnv

def test_environment():
    """Test the basic functionality of the Snake environment."""
    print("🐍 Testing Snake RL Environment...")
    
    # Create environment
    env = SnakeEnv()
    print(f"✅ Environment created successfully")
    print(f"   Action space: {env.action_space}")
    print(f"   Observation space: {env.observation_space}")
    
    # Test reset
    print("\n🔄 Testing reset...")
    obs, info = env.reset()
    print(f"   Initial observation: {obs}")
    print(f"   Snake head: ({obs[0]}, {obs[1]})")
    print(f"   Food: ({obs[2]}, {obs[3]})")
    print(f"   Direction: {obs[4]} (0=UP, 1=DOWN, 2=LEFT, 3=RIGHT)")
    print(f"   Length: {obs[5]}")
    
    # Test a few random actions
    print("\n🎲 Testing random actions...")
    for step in range(5):
        # Take random action
        action = env.action_space.sample()
        print(f"   Step {step + 1}: Action {action}")
        
        # Step the environment
        obs, reward, terminated, truncated, info = env.step(action)
        
        print(f"     Reward: {reward:.2f}")
        print(f"     Score: {info['score']}")
        print(f"     Snake length: {info['snake_length']}")
        print(f"     Distance to food: {info['distance_to_food']:.2f}")
        print(f"     Terminated: {terminated}")
        
        if terminated:
            print("     Game over! Resetting...")
            obs, info = env.reset()
            break
    
    # Test render (this will open a window)
    print("\n🎮 Testing render...")
    print("   A game window should open. Close it to continue.")
    env.render()
    
    # Clean up
    env.close()
    print("\n✅ Environment test completed successfully!")

if __name__ == "__main__":
    test_environment()
