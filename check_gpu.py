from llmark.utils.gpu import get_available_gpu_devices
import requests, json, argparse
from pathlib import Path

def send_alarm(dry_run: bool = False):
    WEBHOOK_URL = "https://discord.com/api/webhooks/1187230163151364106/EiSuWqp9vkBNLxb_hU1V2CA3mv1EANCt4RAtqizEGohN2FTTjzxlYuz8V6i8creeJ_0f"

    if dry_run:
        requests.post(
            WEBHOOK_URL,
            headers={"Content-Type": "application/json"},
            data=json.dumps(
                {
                    "content": "Test Alarm",
                    "tts": False,
                    "embeds": [
                        {
                            "description": "Test Alarm"
                        }
                    ],
                }
            ),
        )
        return

    requests.post(
        WEBHOOK_URL,
        headers={"Content-Type": "application/json"},
        data=json.dumps(
            {
                "content": f"사용 가능한 GPU: {get_available_gpu_devices(raise_error=False)}",
                "tts": False,
                "embeds": [
                    {
                        "description": "서버를 확인하세요."
                    }
                ],
            }
        ),
    )

    with open("./gpu.txt", "w") as f:
        f.write("ok")

def main(args: argparse.Namespace):
    if args.dry_run:
        send_alarm(args.dry_run)
        return
    
    if get_available_gpu_devices(raise_error=False) is not None:
        if not Path("./gpu.txt").exists():
            send_alarm()

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--dry-run", action='store_true', help="Test", dest="dry_run")

    args = parser.parse_args()

    main(args)