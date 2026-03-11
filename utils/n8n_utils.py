import requests


def send_to_n8n(payload: dict, webhook_url: str):
    try:
        response = requests.post(webhook_url, json=payload, timeout=30)
        return {
            "status_code": response.status_code,
            "response_text": response.text
        }
    except Exception as e:
        return {
            "status_code": 500,
            "response_text": str(e)
        }
