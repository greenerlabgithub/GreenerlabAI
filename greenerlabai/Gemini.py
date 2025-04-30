import base64
import google.generativeai as genai
from google.genai import types
import azure.functions as func
import os  # 환경 변수 불러오기

API_KEY = os.getenv("GOOGLE_API_KEY")

def generate(image_data: bytes, additional_info: str):
    client = genai.configure(api_key=API_KEY)

    model = genai.GenerativeModel("gemini-2.0-flash")

    contents = [
        types.Content(
            role="user",
            parts=[
                types.Part.from_text(text=f"""이 이미지는 수목 혹은 식물에 영향을 주는 곤충 혹은 증상이 발현한 병증입니다. {additional_info}"""),
                types.Part.from_image(data=image_data),  # 이미지 데이터 전달
            ],
        ),
    ]
    tools = [
        types.Tool(google_search=types.GoogleSearch()),  # 구글 검색 도구
    ]
    generate_content_config = types.GenerateContentConfig(
        temperature=0.5,
        top_p=1,
        safety_settings=[
            types.SafetySetting(
                category="HARM_CATEGORY_HARASSMENT",
                threshold="BLOCK_ONLY_HIGH",
            ),
            types.SafetySetting(
                category="HARM_CATEGORY_HATE_SPEECH",
                threshold="BLOCK_ONLY_HIGH",
            ),
            types.SafetySetting(
                category="HARM_CATEGORY_SEXUALLY_EXPLICIT",
                threshold="BLOCK_LOW_AND_ABOVE",
            ),
            types.SafetySetting(
                category="HARM_CATEGORY_DANGEROUS_CONTENT",
                threshold="BLOCK_ONLY_HIGH",
            ),
        ],
        tools=tools,
        response_mime_type="text/plain",
        system_instruction=[
            types.Part.from_text(text="""이 이미지들은 수목 혹은 식물에 영향을 주는 곤충 혹은 병증이 발현한 사진입니다.
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
방제 및 처리 방법 :"""),
        ],
    )

    response = ""
    for chunk in client.models.generate_content_stream(
        model=model,
        contents=contents,
        config=generate_content_config,
    ):
        response += chunk.text  # 전체 응답 수집

    return response

def main(req: func.HttpRequest) -> func.HttpResponse:
    try:
        # JSON 바디 파싱
        try:
            body = req.get_json()  # Content-Type: application/json
        except ValueError:
            return func.HttpResponse("잘못된 JSON 형식입니다.", status_code=400)

        image_base64    = body.get("imageData")
        additional_info = body.get("additionalInfo", "")

        if not image_base64:
            return func.HttpResponse("JSON에 imageData 필드가 없습니다.", status_code=400)

        # Base64로 인코딩된 이미지를 디코딩
        image_data = base64.b64decode(image_base64)

        # 이미지 데이터를 전달하여 결과 생성
        result = generate(image_data, additional_info)

        return func.HttpResponse(
            body=result,
            status_code=200,
            mimetype="text/plain"
        )

    except Exception as e:
        return func.HttpResponse(f"Error: {str(e)}", status_code=500)
