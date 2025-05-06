import requests

# Replace with any public image URL
image_url = "https://media.springernature.com/full/springer-static/image/art%3A10.1007%2Fs40747-022-00818-2/MediaObjects/40747_2022_818_Fig13_HTML.png"

payload = {
            "model": "microsoft/Phi-4-multimodal-instruct",
                "messages": [
                            {
                                            "role": "user",
                                                        "content": [
                                                                            {"type": "text", "text": "What does this image show? Explicitly mention the axis information, label information, what can be inferred from the graph."},
                                                                                            {
                                                                                                                    "type": "image_url",
                                                                                                                                        "image_url": {
                                                                                                                                                                    "url": image_url  # no base64 needed here
                                                                                                                                                                                        }
                                                                                                                                                        }
                                                                                                        ]
                                                                }
                                ],
                    "temperature": 0.01
                    }

response = requests.post("http://localhost:8000/v1/chat/completions", json=payload)
print(response.json())

