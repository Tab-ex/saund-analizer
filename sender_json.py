# sender_json.py
import urllib.request
import json

def send_to_server(payload, server_url="http://localhost:8000/api/detect"):
    """
    Отправляет словарь payload на сервер в виде JSON.
    Возвращает True при успехе, False при ошибке.
    """
    data_bytes = json.dumps(payload, ensure_ascii=False).encode("utf-8")
    req = urllib.request.Request(
        server_url,
        data=data_bytes,
        headers={"Content-Type": "application/json"},
        method="POST"
    )
    try:
        with urllib.request.urlopen(req, timeout=10) as response:
            print(f"✅ Сервер принял данные. HTTP {response.status}")
            return True
    except urllib.error.HTTPError as e:
        print(f"⚠️  HTTP Ошибка {e.code}: {e.reason}")
        return False
    except Exception as e:
        print(f"❌ Сетевая ошибка: {e}")
        return False