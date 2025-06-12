import sys
import os
import time
import keyboard

import gymnasium as gym

# Add the parent directory to sys.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import ale_py

# Funktion zur Umwandlung von Tasteneingaben in Aktionen
def get_action_from_keyboard():
    # DonkeyKong Aktionen: 0: NOOP, 1: UP, 2: RIGHT, 3: LEFT, 4: DOWN, 5: UP+RIGHT, 6: UP+LEFT, 7: DOWN+RIGHT, 8: DOWN+LEFT
    if keyboard.is_pressed('up') and keyboard.is_pressed('right'):
        return 5
    elif keyboard.is_pressed('up') and keyboard.is_pressed('left'):
        return 6
    elif keyboard.is_pressed('down') and keyboard.is_pressed('right'):
        return 7
    elif keyboard.is_pressed('down') and keyboard.is_pressed('left'):
        return 8
    elif keyboard.is_pressed('up'):
        return 1
    elif keyboard.is_pressed('right'):
        return 2
    elif keyboard.is_pressed('left'):
        return 3
    elif keyboard.is_pressed('down'):
        return 4
    else:
        return 0  # NOOP

# Drucke Bedienungsanleitung
print("Tastatursteuerung für Donkey Kong:")
print("Pfeiltasten: hoch, runter, links, rechts")
print("Kombinationen wie hoch+links oder runter+rechts sind möglich")
print("Drücken Sie 'q' zum Beenden")

gym.register_envs(ale_py)
env = gym.make('ALE/DonkeyKong-v5', render_mode='human', obs_type='rgb', max_episode_steps=-1)

for episode_idx in range(20):
    state, _ = env.reset()
    total_reward = 0
    print(f"\nEpisode {episode_idx+1} gestartet")

    while True:
        # Beenden, wenn q gedrückt wird
        if keyboard.is_pressed('q'):
            print("Spiel beendet durch Benutzer")
            env.close()
            sys.exit(0)
            
        action = get_action_from_keyboard()
        state, reward, terminated, truncated, info = env.step(action=action)
        print(reward)
        
        total_reward += reward
        
        # if info != {}:
        #     print(info)

        if terminated or truncated:
            print(f"Episode beendet. Gesamtpunktzahl: {total_reward}")
            break
        
        # Kurze Pause, um CPU-Last zu reduzieren
        time.sleep(0.01)

env.close()