import requests
import base64

# Load your image
with open("example.png", "rb") as f:
        image_data = base64.b64encode(f.read()).decode("utf-8")

        payload = {
                    "model": "microsoft/Phi-4-multimodal-instruct",
                        "messages": [
                                    {
                                                    "role": "user",
                                                                "content": [
                                                                                    {"type": "text", "text": "Describe what is shown in this image."},
                                                                                                    {
                                                                                                                            "type": "image_url",
                                                                                                                                                "image_url": {
                                                                                                                                                                            "url": f"data:image/jpeg;base64,{image_data}"
                                                                                                                                                                                                }
                                                                                                                                                                }
                                                                                                                ]
                                                                        }
                                        ],
                            "temperature": 0.5
                            }

        response = requests.post("http://localhost:8000/v1/chat/completions", json=payload)
        print(response.json())

