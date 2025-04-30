import base64
import logging
from io import BytesIO
from PIL import Image
import google.generativeai as genai
from google.genai import types
import azure.functions as func
import os

API_KEY = os.getenv("GOOGLE_API_KEY")
logger = logging.getLogger("greenerlabai")
logger.setLevel(logging.DEBUG)

def part_from_image_bytes(image_data: bytes):
    # Pillow 로 이미지 열기
    img = Image.open(BytesIO(image_data))
    fmt = img.format  # e.g. 'JPEG', 'PNG', 'GIF', ...
    # Image.MIME 에서 해당 포맷의 MIME 타입을 가져오고, 없으면 기본 'image/jpeg'
    mime_type = Image.MIME.get(fmt, "image/jpeg")
    logger.debug("Pillow 감지 이미지 포맷: %s → %s", fmt, mime_type)
    return types.Part.from_bytes(data=image_data, mime_type=mime_type)

def generate(image_data: bytes, additional_info: str):
    client = genai.configure(api_key=API_KEY)
    model = genai.GenerativeModel("gemini-2.0-flash")

    contents = [
        types.Content(
            role="user",
            parts=[
                types.Part.from_text(text=(
                    "이 이미지는 수목 혹은 식물에 영향을 주는 곤충 혹은 증상이 발현한 병증입니다. "
                    f"{additional_info}"
                )),
                # Pillow 기반 파트 생성
                part_from_image_bytes(image_data),
            ],
        ),
    ]
    tools = [ types.Tool(google_search=types.GoogleSearch()) ]
    config = types.GenerateContentConfig(
        temperature=0.5,
        top_p=1,
        safety_settings=[
            types.SafetySetting(category="HARM_CATEGORY_HARASSMENT",     threshold="BLOCK_ONLY_HIGH"),
            types.SafetySetting(category="HARM_CATEGORY_HATE_SPEECH",    threshold="BLOCK_ONLY_HIGH"),
            types.SafetySetting(category="HARM_CATEGORY_SEXUALLY_EXPLICIT", threshold="BLOCK_LOW_AND_ABOVE"),
            types.SafetySetting(category="HARM_CATEGORY_DANGEROUS_CONTENT", threshold="BLOCK_ONLY_HIGH"),
        ],
        tools=tools,
        response_mime_type="text/plain",
        system_instruction=[
            types.Part.from_text(text=("""이 이미지들은 수목 혹은 식물에 영향을 주는 곤충 혹은 증상이 발현한 병증입니다.
이미지에서 보이는 곤충 혹은 병증을 분석 및 추출하여 검색엔진에서 가장 유사한 정보를 찾아냅니다.
먼저 Google Search를 통해 가장 유사한 이미지 혹은 오브젝트를 찾아 정보를 추출합니다.
유사한 이미지 혹은 오브젝트를 찾지 못할 경우 예상되는 리스트를 알려줍니다.
기본적인 정보는 아래 폼과 같이 전달됩니다.
수목 혹은 식물 : 
촬영된 부위 : 
현재 증상 : 
가장 유사한 정보를 찾을 경우 아래 폼과 같이 정리해줍니다.
병해충 혹은 증상 :
병해충 혹은 증상의 자세한 정보 :
방제 및 처리 방법 :"""
            )),
        ],
    )

    response =  client.models.generate_content(
        model=model,
        contents=contents,
        config=config
    )

    return response.text

def main(req: func.HttpRequest, context: func.Context) -> func.HttpResponse:
    logger.info("Invocation ID=%s: 요청 수신", context.invocation_id)
    try:
        body = req.get_json()
        image_base64    = body.get("imageData")
        additional_info = body.get("additionalInfo", "")

        if not image_base64:
            return func.HttpResponse("JSON에 imageData 필드가 없습니다.", status_code=400)

        image_data = base64.b64decode(image_base64)
        result = generate(image_data, additional_info)

        return func.HttpResponse(body=result, status_code=200, mimetype="text/plain")

    except ValueError:
        return func.HttpResponse("잘못된 JSON 형식입니다.", status_code=400)
    except Exception as e:
        logger.exception("처리 중 예외 발생")
        return func.HttpResponse(f"Error: {str(e)}", status_code=500)
