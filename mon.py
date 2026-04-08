#!/usr/bin/env python3
"""GPU空闲邮件通知 - 最简版"""
import subprocess, time, smtplib, argparse
from email.mime.text import MIMEText

parser = argparse.ArgumentParser()
parser.add_argument("--from", dest="sender", required=True)
parser.add_argument("--password", required=True)
parser.add_argument("--to", required=True)
parser.add_argument("--min", type=int, default=4)
parser.add_argument("--interval", type=int, default=60)
parser.add_argument("--threshold", type=int, default=5)
args = parser.parse_args()

last_alert = 0

while True:
    out = subprocess.run(
        ["nvidia-smi", "--query-gpu=index,utilization.gpu", "--format=csv,noheader,nounits"],
        capture_output=True, text=True
    ).stdout.strip()

    idle = [l for l in out.split("\n") if l and int(l.split(",")[1].strip()) < args.threshold]
    free = len(idle)

    ts = time.strftime("%H:%M:%S")
    print(f"[{ts}] Total GPUs: {len(out.split(chr(10)))}, Free: {free}", flush=True)

    if free >= args.min and time.time() - last_alert >= 600:
        body = f"{free} GPU(s) are free now!\nTime: {time.strftime('%Y-%m-%d %H:%M:%S')}"
        msg = MIMEText(body)
        msg["Subject"] = f"[GPU Alert] {free} GPU(s) Available!"
        msg["From"] = args.sender
        msg["To"] = args.to

        with smtplib.SMTP("smtp.163.com", 25) as s:
            s.login(args.sender, args.password)
            s.send_message(msg)
        last_alert = time.time()
        print(f"  [ALERT] Email sent!", flush=True)

    time.sleep(args.interval)