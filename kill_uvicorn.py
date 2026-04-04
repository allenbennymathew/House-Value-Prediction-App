import subprocess

output = subprocess.check_output("netstat -ano", shell=True).decode()
for line in output.splitlines():
    if "0.0.0.0:8000" in line and "LISTENING" in line:
        pid = line.strip().split()[-1]
        print(f"Killing PID {pid}")
        subprocess.run(f"taskkill /F /PID {pid}", shell=True)
