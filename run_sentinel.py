import subprocess
import sys
import time

def main():
    print("==================================================")
    print("🚀 STARTING SENTINEL 2.0 (THREE-TIER ARCHITECTURE)")
    print("==================================================")

    processes = []
    try:
        print("[1/3] Starting Host Server (UI & Database)...")
        p1 = subprocess.Popen([sys.executable, "host_server.py"])
        processes.append(("Host Server", p1))
        time.sleep(2)

        print("[2/3] Starting Gemma Orchestrator & Audio Engine...")
        p2 = subprocess.Popen([sys.executable, "main.py"])
        processes.append(("Gemma Orchestrator", p2))
        time.sleep(2)

        print("[3/3] Starting Vision Reflex Loop (YOLO Pose + Privacy)...")
        p3 = subprocess.Popen([sys.executable, "vision_trigger.py", "--source", "Fall2_Cam4_with_audio.mp4"])
        processes.append(("Vision Trigger", p3))
        
        print("\n==================================================")
        print("✅ ALL SYSTEMS ONLINE!")
        print("🌍 Open your browser to: http://127.0.0.1:8000")
        print("🛑 Press Ctrl+C in this terminal to shut everything down.")
        print("==================================================\n")

        while True:
            time.sleep(1)

    except KeyboardInterrupt:
        print("\nShutting down all Sentinel services...")
    finally:
        for name, p in processes:
            print(f"Terminating {name}...")
            p.terminate()
            p.wait()
        print("Shutdown complete. Goodbye!")

if __name__ == "__main__":
    main()
