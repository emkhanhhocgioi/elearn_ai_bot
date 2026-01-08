from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from openai import OpenAI
import json
import re
import os
from dotenv import load_dotenv
import requests
import base64
import cloudinary
import cloudinary.api
import cloudinary.uploader

# Load environment variables from .env file
load_dotenv()

app = FastAPI()


cloudinary.config(
    cloud_name=os.getenv("cloud_name"),
    api_key=os.getenv("api_key"),
    api_secret=os.getenv("api_secret"),
    secure=True
)

# Global Rubrics for grading
GLOBAL_RUBRICS = {
    "Ngữ văn": [
        {"name": "Đọc hiểu văn bản", "weight": 30},
        {"name": "Viết (cấu trúc, lập luận)", "weight": 40},
        {"name": "Diễn đạt & dùng từ", "weight": 20},
        {"name": "Chính tả, ngữ pháp", "weight": 10}
    ],
    "Toán": [
        {"name": "Đáp án đúng ", "weight": 90},
        {"name": "Kỹ năng tính toán", "weight": 10},
     
    ],
    "Tiếng Anh": [
        {"name": "Từ vựng – ngữ pháp", "weight": 30},
        {"name": "Nghe", "weight": 20},
        {"name": "Nói", "weight": 20},
        {"name": "Đọc – Viết", "weight": 30}
    ],
    "Vật lí": [
        {"name": "Hiểu khái niệm, định luật", "weight": 35},
        {"name": "Vận dụng giải bài tập", "weight": 35},
        {"name": "Thí nghiệm – quan sát", "weight": 20},
        {"name": "Trình bày", "weight": 10}
    ],
    "Hóa học": [
        {"name": "Kiến thức hóa học", "weight": 40},
        {"name": "Phương trình, tính toán", "weight": 30},
        {"name": "Thực hành – an toàn", "weight": 20},
        {"name": "Trình bày", "weight": 10}
    ],
    "Sinh học": [
        {"name": "Hiểu kiến thức sinh học", "weight": 40},
        {"name": "Vận dụng thực tiễn", "weight": 25},
        {"name": "Quan sát – phân tích", "weight": 25},
        {"name": "Thuật ngữ khoa học", "weight": 10}
    ],
    "Lịch sử": [
        {"name": "Sự kiện – mốc thời gian", "weight": 40},
        {"name": "Phân tích – nhận xét", "weight": 30},
        {"name": "Liên hệ thực tế", "weight": 20},
        {"name": "Trình bày", "weight": 10}
    ],
    "Địa lí": [
        {"name": "Kiến thức địa lí", "weight": 35},
        {"name": "Bản đồ – biểu đồ", "weight": 30},
        {"name": "Phân tích số liệu", "weight": 25},
        {"name": "Trình bày", "weight": 10}
    ],
    "Giáo dục công dân": [
        {"name": "Đạo đức – pháp luật", "weight": 40},
        {"name": "Xử lý tình huống", "weight": 30},
        {"name": "Thái độ, hành vi", "weight": 20},
        {"name": "Trình bày", "weight": 10}
    ],
    "Công nghệ": [
        {"name": "Kiến thức công nghệ", "weight": 30},
        {"name": "Kỹ năng thực hành", "weight": 40},
        {"name": "Sản phẩm / quy trình", "weight": 20},
        {"name": "An toàn, kỷ luật", "weight": 10}
    ],
    "Tin học": [
        {"name": "Kiến thức tin học", "weight": 30},
        {"name": "Thao tác máy tính", "weight": 40},
        {"name": "Tư duy thuật toán", "weight": 20},
        {"name": "Ý thức sử dụng CNTT", "weight": 10}
    ],
    "Thể dục": [
        {"name": "Kỹ thuật động tác", "weight": 40},
        {"name": "Thể lực", "weight": 30},
        {"name": "Ý thức luyện tập", "weight": 20},
        {"name": "Kỷ luật", "weight": 10}
    ]
}

# Subject mapping from English to Vietnamese
SUBJECT_MAPPING = {
    "math": "Toán",
    "van": "Ngữ văn",
    "english": "Tiếng Anh",
    "physics": "Vật lí",
    "chemistry": "Hóa học",
    "biology": "Sinh học",
    "geography": "Địa lí",
    "history": "Lịch sử",
    "civics": "Giáo dục công dân",
    "informatics": "Tin học"
}


# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize OpenAI client
print("Initializing OpenAI client...")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
client = OpenAI(api_key=OPENAI_API_KEY)
MODEL_NAME = "gpt-4o-mini"  # Sử dụng GPT-4o mini cho hiệu quả
print("OpenAI client initialized")

def extract_json_from_text(text):
    """
    Hàm helper để extract JSON từ text response.
    Thử nhiều pattern khác nhau để tìm JSON hợp lệ.
    """
    # Thử 1: Parse trực tiếp
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass
    
    # Thử 2: Tìm JSON trong code block (```json...```)
    json_match = re.search(r'```json\s*([\[\{].*?[\]\}])\s*```', text, re.DOTALL)
    if json_match:
        try:
            return json.loads(json_match.group(1))
        except json.JSONDecodeError:
            pass
    
    # Thử 3: Tìm JSON array [...]
    array_match = re.search(r'\[\s*\{.*?\}\s*(?:,\s*\{.*?\}\s*)*\]', text, re.DOTALL)
    if array_match:
        try:
            return json.loads(array_match.group(0))
        except json.JSONDecodeError:
            pass
    
    # Thử 4: Tìm JSON object {...}
    object_match = re.search(r'\{[^{}]*(?:\{[^{}]*\}[^{}]*)*\}', text, re.DOTALL)
    if object_match:
        try:
            return json.loads(object_match.group(0))
        except json.JSONDecodeError:
            pass
    
    # Thử 5: Xử lý trường hợp có trailing comma
    cleaned_text = re.sub(r',\s*(\]|\})', r'\1', text)
    try:
        return json.loads(cleaned_text)
    except json.JSONDecodeError:
        pass
    
    return None

def readFileFromUrl(url: str) -> str:
    response = requests.get(url)
    response.raise_for_status()  # báo lỗi nếu URL sai
    
    return response.text

def getUrlFileFormat(url: str) -> str:
    result = cloudinary.api.resource(url)
    return result.get("format", "")
    
class PromptRequest(BaseModel):
    prompt: str
    task: str = "question_generate_van"  # Default task
    max_tokens: int = 256
    temperature: float = 0.7
    top_p: float = 0.9

class GradingRequest(BaseModel):
    exercise_question: str
    subject: str
    student_answer: str
    
class BaseOnRecentTestRequest(BaseModel):
    recent_tests: list[dict]
    questionTypes: list[str]
    subject: str
    
    

class GenerateQuestionRequest(BaseModel):
    prompt: str
    subject: str  # math, van, english, physics, chemistry, biology, geography, history, civics, informatics

class TeacherFeedbackRequest(BaseModel):
    teacher_comment: str | list[str]  # Accept both string and list
    subject: str
    lesson: str
    test_answers: list[dict]

class AutoGradingRequest(BaseModel):
    exercise_question: str
    fileUrl: str
    subject: str  

class PerformanceQuestionRequest(BaseModel):
    subject: str
    recent_tests: list[dict]  # List of test info with subject, title, score, submissionTime

# Request model for recent test grading
class RecentTestGradingRequest(BaseModel):
    subject: str
    questions: list[dict]  # List of {question, student_answer, topic, difficulty}

# Request model for rubric-based grading
class RubricGradingRequest(BaseModel):
    test_title: str
    subject: str
    questions_and_answers: list[dict]  # List of {question, questionType, solution, grade, studentAnswer, isCorrect}
    rubric_criteria: list[dict]  # List of {name, weight, description?}
    student_name: str = "Học sinh"
    
@app.get("/")
def read_root():
    return {"message": "Văn học AI API", "status": "running"}

@app.post("/generate")
async def generate_response(request: PromptRequest):
    try:
        response = client.chat.completions.create(
            model=MODEL_NAME,
            messages=[
                {"role": "system", "content": "Bạn là trợ lý AI chuyên về văn học Việt Nam."},
                {"role": "user", "content": request.prompt}
            ],
            max_tokens=request.max_tokens,
            temperature=request.temperature,
            top_p=request.top_p
        )
        
        response_text = response.choices[0].message.content
        
        # Parse JSON từ response
        parsed_response = extract_json_from_text(response_text)
        if parsed_response is None:
            parsed_response = response_text  # Nếu không parse được, giữ nguyên text
        
        return {
            "success": True,
            "response": parsed_response,
            "prompt": request.prompt
        }
        
    except Exception as e:
        return {
            "success": False,
            "error": str(e)
        }

@app.post("/generate_question")
async def generate_question(request: GenerateQuestionRequest):
    """
    Tạo câu hỏi cho bất kỳ môn học THCS nào
    Subjects: math, van, english, physics, chemistry, biology, geography, history, civics, informatics
    """
    # Định nghĩa thông tin cho từng môn học
    subject_config = {
        "math": {
            "name": "Toán",
            "system_prompt": "Bạn là giáo viên Toán THCS. CHỈ trả về JSON, không có text khác.",
            "example": {
                "question": "Cho A = {1,2,3} và B = {2,3,4}. Tìm A giao B.",
                "answer": "A giao B là tập hợp các phần tử thuộc cả A và B. Các phần tử chung là 2 và 3. Vậy A giao B = {2,3}.",
                "difficulty": "easy"
            }
        },
        "van": {
            "name": "Ngữ văn",
            "system_prompt": "Bạn là giáo viên Ngữ văn THCS. CHỈ trả về JSON, không có text khác.",
            "example": {
                "question": "Phân tích hình ảnh người lái đò trong tác phẩm 'Người lái đò sông Đà' của Nguyễn Tuân.",
                "answer": "Người lái đò là hình ảnh người lao động chân chất, giản dị. Tác giả miêu tả ông qua ngoại hình, cử chỉ và lời nói, thể hiện sức mạnh và tình yêu nghề nghiệp. Hình ảnh này ca ngợi vẻ đẹp của người lao động Việt Nam.",
                "difficulty": "medium"
            }
        },
        "english": {
            "name": "Tiếng Anh",
            "system_prompt": "Bạn là giáo viên Tiếng Anh THCS. CHỈ trả về JSON, không có text khác.",
            "example": {
                "question": "Fill in the blank: She _____ to school every day. (go/goes)",
                "answer": "Đáp án: goes. Giải thích: Chủ ngữ 'She' là ngôi thứ 3 số ít nên động từ phải thêm 's/es'. Do đó ta dùng 'goes'.",
                "difficulty": "easy"
            }
        },
        "physics": {
            "name": "Vật lý",
            "system_prompt": "Bạn là giáo viên Vật lý THCS. CHỈ trả về JSON, không có text khác.",
            "example": {
                "question": "Một vật chuyển động đều với vận tốc 36 km/h trong 2 giờ. Tính quãng đường vật đi được.",
                "answer": "Quãng đường = vận tốc × thời gian = 36 km/h × 2h = 72 km. Vậy quãng đường vật đi được là 72 km.",
                "difficulty": "easy"
            }
        },
        "chemistry": {
            "name": "Hóa học",
            "system_prompt": "Bạn là giáo viên Hóa học THCS. CHỈ trả về JSON, không có text khác.",
            "example": {
                "question": "Viết phương trình hóa học của phản ứng giữa natri (Na) và nước (H₂O).",
                "answer": "Phương trình: 2Na + 2H₂O → 2NaOH + H₂. Giải thích: Natri là kim loại kiềm hoạt động mạnh, phản ứng với nước tạo dung dịch bazơ natri hydroxit và giải phóng khí hydro.",
                "difficulty": "medium"
            }
        },
        "biology": {
            "name": "Sinh học",
            "system_prompt": "Bạn là giáo viên Sinh học THCS. CHỈ trả về JSON, không có text khác.",
            "example": {
                "question": "Nêu chức năng chính của hệ tuần hoàn ở động vật có xương sống.",
                "answer": "Hệ tuần hoàn có các chức năng: 1) Vận chuyển oxy và chất dinh dưỡng đến các tế bào, 2) Đưa CO₂ và chất thải ra khỏi cơ thể, 3) Điều hòa thân nhiệt, 4) Bảo vệ cơ thể chống bệnh tật.",
                "difficulty": "easy"
            }
        },
        "geography": {
            "name": "Địa lý",
            "system_prompt": "Bạn là giáo viên Địa lý THCS. CHỈ trả về JSON, không có text khác.",
            "example": {
                "question": "Nêu đặc điểm khí hậu nhiệt đới gió mùa ở Việt Nam.",
                "answer": "Khí hậu nhiệt đới gió mùa có đặc điểm: 1) Nhiệt độ cao quanh năm (trung bình >20°C), 2) Có 2 mùa rõ rệt: mùa mưa và mùa khô, 3) Lượng mưa lớn tập trung vào mùa hè, 4) Chịu ảnh hưởng của gió mùa Đông Bắc và Tây Nam.",
                "difficulty": "easy"
            }
        },
        "history": {
            "name": "Lịch sử",
            "system_prompt": "Bạn là giáo viên Lịch sử THCS. CHỈ trả về JSON, không có text khác.",
            "example": {
                "question": "Nêu ý nghĩa lịch sử của chiến thắng Bạch Đằng năm 938.",
                "answer": "Ý nghĩa: 1) Đánh bại quân Nam Hán, kết thúc 1000 năm Bắc thuộc, 2) Mở ra thời kỳ độc lập tự chủ cho dân tộc Việt Nam, 3) Khẳng định ý chí tự chủ và năng lực quân sự của dân tộc, 4) Ngô Quyền trở thành vua đầu tiên của nước Việt Nam độc lập.",
                "difficulty": "medium"
            }
        },
        "civics": {
            "name": "Giáo dục Công dân",
            "system_prompt": "Bạn là giáo viên Giáo dục Công dân THCS. CHỈ trả về JSON, không có text khác.",
            "example": {
                "question": "Nêu quyền và nghĩa vụ cơ bản của công dân Việt Nam.",
                "answer": "Quyền: 1) Quyền bình đẳng, 2) Quyền tự do ngôn luận, 3) Quyền bầu cử và ứng cử, 4) Quyền được học tập. Nghĩa vụ: 1) Tuân thủ pháp luật, 2) Bảo vệ Tổ quốc, 3) Nộp thuế, 4) Giữ gìn môi trường.",
                "difficulty": "easy"
            }
        },
        "informatics": {
            "name": "Tin học",
            "system_prompt": "Bạn là giáo viên Tin học THCS. CHỈ trả về JSON, không có text khác.",
            "example": {
                "question": "Viết chương trình Python tính tổng hai số a và b.",
                "answer": "Code:\na = int(input('Nhập số a: '))\nb = int(input('Nhập số b: '))\ntong = a + b\nprint('Tổng =', tong)\n\nGiải thích: Chương trình nhận 2 số từ người dùng, cộng lại và in kết quả.",
                "difficulty": "easy"
            }
        }
    }
    
    # Kiểm tra subject hợp lệ
    if request.subject not in subject_config:
        return {
            "success": False,
            "error": f"Môn học '{request.subject}' không hợp lệ. Các môn học hỗ trợ: {', '.join(subject_config.keys())}"
        }
    
    try:
        config = subject_config[request.subject]
        
        prompt = f"""Tạo câu hỏi {config['name']} theo yêu cầu: {request.prompt}

Trả về JSON với format SAU (KHÔNG thêm text khác):
{{"question": "câu hỏi", "answer": "lời giải chi tiết", "difficulty": "easy"}}

Ví dụ:
{json.dumps(config['example'], ensure_ascii=False)}"""
        
        response = client.chat.completions.create(
            model=MODEL_NAME,
            messages=[
                {"role": "system", "content": config['system_prompt']},
                {"role": "user", "content": prompt}
            ],
            max_tokens=1024,
            temperature=0.3,
            top_p=0.8,
            response_format={"type": "json_object"}
        )
        
        response_text = response.choices[0].message.content
        result = extract_json_from_text(response_text)
        
        # Validate JSON structure
        if result and isinstance(result, dict):
            if "question" in result and "answer" in result:
                if isinstance(result.get("question"), str) and isinstance(result.get("answer"), str):
                    if "difficulty" not in result:
                        result["difficulty"] = "medium"
                    
                    return {
                        "success": True,
                        "result": result,
                        "subject": request.subject,
                        "prompt": request.prompt
                    }
        
        return {
            "success": False,
            "error": "Model không tạo được JSON hợp lệ. Vui lòng thử lại.",
            "prompt": request.prompt,
            "subject": request.subject
        }
    
    except Exception as e:
        return {
            "success": False,
            "error": str(e)
        }


@app.post('/auto-grading')
async def auto_grading(request: GradingRequest):
    """
    Tự động chấm điểm bài tập cho các môn THCS
    Subjects: math, van, english, physics, chemistry, biology, geography, history, civics, informatics
    """
    subject_grading_prompts = {
        "math": "Bạn là giáo viên Toán THCS. Hãy chấm điểm bài làm toán dựa trên các bước giải và kết quả cuối cùng.",
        "van": "Bạn là giáo viên Ngữ văn THCS. Hãy chấm điểm bài làm văn dựa trên nội dung, lập luận và diễn đạt.",
        "english": "Bạn là giáo viên Tiếng Anh THCS. Hãy chấm điểm bài làm tiếng Anh dựa trên ngữ pháp, từ vựng và ý tưởng.",
        "physics": "Bạn là giáo viên Vật lý THCS. Hãy chấm điểm bài làm vật lý dựa trên phương pháp giải và kết quả.",
        "chemistry": "Bạn là giáo viên Hóa học THCS. Hãy chấm điểm bài làm hóa học dựa trên các phản ứng và tính toán.",
        "biology": "Bạn là giáo viên Sinh học THCS. Hãy chấm điểm bài làm sinh học dựa trên kiến thức và phân tích.",
        "geography": "Bạn là giáo viên Địa lý THCS. Hãy chấm điểm bài làm địa lý dựa trên hiểu biết về địa hình và khí hậu.",
        "history": "Bạn là giáo viên Lịch sử THCS. Hãy chấm điểm bài làm lịch sử dựa trên sự kiện và phân tích lịch sử.",
        "civics": "Bạn là giáo viên Giáo dục Công dân THCS. Hãy chấm điểm bài làm GDCD dựa trên quyền và nghĩa vụ công dân.",
        "informatics": "Bạn là giáo viên Tin học THCS. Hãy chấm điểm bài làm tin học dựa trên code và logic."
    }

    if request.subject not in subject_grading_prompts:
        return {
            "success": False,
            "error": f"Môn học '{request.subject}' không hợp lệ. Các môn học hỗ trợ: {', '.join(subject_grading_prompts.keys())}"
        }

    try:
        grading_prompt = subject_grading_prompts[request.subject]
        
        # Get rubric for the subject
        subject_vietnamese = SUBJECT_MAPPING.get(request.subject, "")
        rubric_criteria = GLOBAL_RUBRICS.get(subject_vietnamese, [])
        
        rubric_text = ""
        if rubric_criteria:
            rubric_text = "\n\nTiêu chí chấm điểm (Rubric):\n"
            for criterion in rubric_criteria:
                rubric_text += f"- {criterion['name']}: {criterion['weight']}%\n"

        prompt = f"""{grading_prompt}

Đề bài: {request.exercise_question}

Bài làm của học sinh:
{request.student_answer}{rubric_text}

Hãy đưa ra điểm số từ 0-10 và nhận xét chi tiết về bài làm dựa trên các tiêu chí rubric.

Trả về format JSON:
{{"isCorrect": <true || false>, "comments": "<nhận xét chi tiết về bài làm theo từng tiêu chí rubric>", "score": <điểm số từ 0-10>}}
"""

        response = client.chat.completions.create(
            model=MODEL_NAME,
            messages=[
                {"role": "system", "content": grading_prompt},
                {"role": "user", "content": prompt}
            ],
            max_tokens=512,
            temperature=0.3,
            top_p=0.9,
            response_format={"type": "json_object"}
        )

        response_text = response.choices[0].message.content
        grading_result = extract_json_from_text(response_text)

        return {
            "success": True,
            "grading_response": grading_result if grading_result is not None else response_text,
            "exercise_question": request.exercise_question,
            "subject": request.subject
        }

    except Exception as e:
        return {
            "success": False,
            "error": str(e)
        }
    
@app.post('/auto-grading/file')
async def auto_grading(request: AutoGradingRequest):
    """
    Tự động chấm điểm bài tập từ file URL cho các môn THCS
    Subjects: math, van, english, physics, chemistry, biology, geography, history, civics, informatics
    """
    subject_grading_prompts = {
        "math": "Bạn là giáo viên Toán THCS. Hãy chấm điểm bài làm toán dựa trên các bước giải và kết quả cuối cùng.",
        "van": "Bạn là giáo viên Ngữ văn THCS. Hãy chấm điểm bài làm văn dựa trên nội dung, lập luận và diễn đạt.",
        "english": "Bạn là giáo viên Tiếng Anh THCS. Hãy chấm điểm bài làm tiếng Anh dựa trên ngữ pháp, từ vựng và ý tưởng.",
        "physics": "Bạn là giáo viên Vật lý THCS. Hãy chấm điểm bài làm vật lý dựa trên phương pháp giải và kết quả.",
        "chemistry": "Bạn là giáo viên Hóa học THCS. Hãy chấm điểm bài làm hóa học dựa trên các phản ứng và tính toán.",
        "biology": "Bạn là giáo viên Sinh học THCS. Hãy chấm điểm bài làm sinh học dựa trên kiến thức và phân tích.",
        "geography": "Bạn là giáo viên Địa lý THCS. Hãy chấm điểm bài làm địa lý dựa trên hiểu biết về địa hình và khí hậu.",
        "history": "Bạn là giáo viên Lịch sử THCS. Hãy chấm điểm bài làm lịch sử dựa trên sự kiện và phân tích lịch sử.",
        "civics": "Bạn là giáo viên Giáo dục Công dân THCS. Hãy chấm điểm bài làm GDCD dựa trên quyền và nghĩa vụ công dân.",
        "informatics": "Bạn là giáo viên Tin học THCS. Hãy chấm điểm bài làm tin học dựa trên code và logic."
    }

    if request.subject not in subject_grading_prompts:
        return {
            "success": False,
            "error": f"Môn học '{request.subject}' không hợp lệ. Các môn học hỗ trợ: {', '.join(subject_grading_prompts.keys())}"
        }

    try:
        file_content = readFileFromUrl(request.fileUrl)
        grading_prompt = subject_grading_prompts[request.subject]
        
        # Get rubric for the subject
        subject_vietnamese = SUBJECT_MAPPING.get(request.subject, "")
        rubric_criteria = GLOBAL_RUBRICS.get(subject_vietnamese, [])
        
        rubric_text = ""
        if rubric_criteria:
            rubric_text = "\n\nTiêu chí chấm điểm (Rubric):\n"
            for criterion in rubric_criteria:
                rubric_text += f"- {criterion['name']}: {criterion['weight']}%\n"

        prompt = f"""{grading_prompt}
Nội dung bài làm:
{file_content}

Hãy kiểm tra xem đúng hay sai so với đề bài: {request.exercise_question}{rubric_text}

YÊU CẦU CHẤM:
- So sánh kết quả và lập luận của bài làm với đề bài.
- Áp dụng các tiêu chí rubric để đánh giá toàn diện.
- Nếu kết luận cuối cùng đúng về mặt toán học thì coi là ĐÚNG,
  kể cả khi cách trình bày khác, thiếu lời giải chi tiết, hoặc dùng từ khác.
- Chỉ trả về isCorrect = false nếu:
  + Kết quả cuối cùng sai, HOẶC
  + Lập luận mâu thuẫn với định nghĩa/toán học cơ bản.
- Không đánh giá dựa trên hình thức, chính tả, hoặc cách diễn đạt.
- Nếu bài làm đúng bản chất toán học → isCorrect = true.

TRẢ VỀ DUY NHẤT JSON (không giải thích thêm ngoài comments):
{{"isCorrect": <true || false>, "comments": "<nhận xét chi tiết về bài làm theo từng tiêu chí rubric>"}}
"""

        response = client.chat.completions.create(
            model=MODEL_NAME,
            messages=[
                {"role": "system", "content": grading_prompt},
                {"role": "user", "content": prompt}
            ],
            max_tokens=512,
            temperature=0.3,
            top_p=0.9
        )

        response_text = response.choices[0].message.content
        grading_result = extract_json_from_text(response_text)

        return {
            "success": True,
            "grading_response": grading_result if grading_result is not None else response_text,
            "exercise_question": request.exercise_question,
            "student_answer" : file_content,
        }

    except Exception as e:
        return {
            "success": False,
            "error": str(e)
        }


@app.post("/auto-grading/image")
async def autograding_image(request: AutoGradingRequest):
    print(request)
    """
    Tự động chấm điểm bài tập từ hình ảnh cho các môn THCS
    Subjects: math, van, english, physics, chemistry, biology, geography, history, civics, informatics
    """
    try:
        # Tải ảnh từ URL và upload lên Cloudinary để lấy URL công khai
        upload_result = cloudinary.uploader.upload(request.fileUrl)
        image_url = upload_result.get("secure_url")

        # Get rubric for the subject
        subject_vietnamese = SUBJECT_MAPPING.get(request.subject, "")
        rubric_criteria = GLOBAL_RUBRICS.get(subject_vietnamese, [])
        
        rubric_text = ""
        if rubric_criteria:
            rubric_text = "\n\nTiêu chí chấm điểm (Rubric):\n"
            for criterion in rubric_criteria:
                rubric_text += f"- {criterion['name']}: {criterion['weight']}%\n"
        
        # Tạo prompt text cho vision API
        grading_text = f"""Bạn là giáo viên {request.subject} THCS. Hãy chấm điểm bài làm dựa trên nội dung trong hình ảnh.

Đề bài: {request.exercise_question}{rubric_text}

YÊU CẦU CHẤM:
- Đọc và phân tích bài làm của học sinh trong hình ảnh
- So sánh kết quả và lập luận của bài làm với đề bài
- Áp dụng các tiêu chí rubric để đánh giá toàn diện
- Nếu kết luận cuối cùng đúng về mặt toán học thì coi là ĐÚNG,
  kể cả khi cách trình bày khác, thiếu lời giải chi tiết, hoặc dùng từ khác
- Chỉ trả về isCorrect = false nếu:
  + Kết quả cuối cùng sai, HOẶC
  + Lập luận mâu thuẫn với định nghĩa/toán học cơ bản
- Không đánh giá dựa trên hình thức, chính tả, hoặc cách diễn đạt
- Nếu bài làm đúng bản chất toán học → isCorrect = true

TRẢ VỀ DUY NHẤT JSON (không giải thích thêm ngoài comments):
{{"isCorrect": <true || false>, "comments": "<nhận xét chi tiết về bài làm theo từng tiêu chí rubric>"}}
"""

        # Sử dụng Vision API với content array để gửi cả text và image
        response = client.chat.completions.create(
            model="gpt-4o-mini",  # gpt-4o-mini hỗ trợ vision
            messages=[
                {
                    "role": "system", 
                    "content": f"Bạn là giáo viên {request.subject} THCS có khả năng đọc và phân tích hình ảnh bài làm của học sinh."
                },
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "text",
                            "text": grading_text
                        },
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": image_url
                            }
                        }
                    ]
                }
            ],
            max_tokens=512,
            temperature=0.3,
            top_p=0.9
        )

        response_text = response.choices[0].message.content
        grading_result = extract_json_from_text(response_text)

        return {
            "success": True,
            "grading_response": grading_result if grading_result is not None else response_text,
            "exercise_question": request.exercise_question,
            "student_answer_image_url": image_url
        }
    except Exception as e:
        return {
            "success": False,
            "error": str(e)
        }
    
@app.get("/health")
def health_check():
    return {
        "status": "healthy",
        "api": "OpenAI",
        "model": MODEL_NAME,
        "client_initialized": client is not None
    }
    

@app.post("/grade-essay")
async def grade_essay(request: GradingRequest):
    """
    Chấm điểm bài văn của học sinh
    """
    try:
        prompt = f"""Hãy chấm điểm bài làm văn theo thang điểm 10 và đưa ra nhận xét cụ thể về ưu điểm và hạn chế của bài viết.

Đề bài: {request.exercise_question}

Bài làm của học sinh:
{request.student_answer}

Hãy đánh giá theo các tiêu chí:
- Nội dung (40%)
- Phân tích & lập luận (30%)
- Diễn đạt & ngôn ngữ (20%)
- Sáng tạo (10%)

Trả về kết quả dưới dạng JSON với format:
{{
    "grade": <điểm số từ 0-10>,
    "comments": "<nhận xét chi tiết về bài làm>",
    "criteria_scores": {{
        "Nội dung": <điểm từ 0-10>,
        "Phân tích & lập luận": <điểm từ 0-10>,
        "Diễn đạt & ngôn ngữ": <điểm từ 0-10>,
        "Sáng tạo": <điểm từ 0-10>
    }},
    "strengths": "<điểm mạnh của bài làm>",
    "weaknesses": "<điểm yếu và hướng cải thiện>"
}}"""
        
        response = client.chat.completions.create(
            model=MODEL_NAME,
            messages=[
                {"role": "system", "content": "Bạn là trợ lý AI chuyên về văn học Việt Nam."},
                {"role": "user", "content": prompt}
            ],
            max_tokens=512,
            temperature=0.3,
            top_p=0.9,
            response_format={"type": "json_object"}
        )
        
        response_text = response.choices[0].message.content
        
        # Parse JSON từ response
        grading_result = extract_json_from_text(response_text)
        
        if grading_result:
            return {
                "success": True,
                "result": grading_result,
                "exercise_question": request.exercise_question
            }
        else:
            return {
                "success": True,
                "result": {
                    "raw_response": response_text
                },
                "exercise_question": request.exercise_question
            }
    
    except Exception as e:
        return {
            "success": False,
            "error": str(e)
        }

@app.post("/recent-test")
async def recent_test(request: BaseOnRecentTestRequest):
    """
    Tạo câu hỏi dựa trên các chủ đề/bài kiểm tra gần đây cho tất cả các môn học THCS
    Subjects: math, van, english, physics, chemistry, biology, geography, history, civics, informatics
    """
    # Định nghĩa config cho từng môn học
    subject_config = {
        "math": {
            "name": "Toán",
            "system_prompt": "Bạn là giáo viên Toán THCS chuyên tạo câu hỏi luyện tập. Hãy luôn tạo đầy đủ số lượng câu hỏi theo yêu cầu.",
            "question_type": "toán học"
        },
        "van": {
            "name": "Ngữ văn",
            "system_prompt": "Bạn là giáo viên Ngữ văn THCS chuyên tạo câu hỏi luyện tập. Hãy luôn tạo đầy đủ số lượng câu hỏi theo yêu cầu.",
            "question_type": "văn học"
        },
        "english": {
            "name": "Tiếng Anh",
            "system_prompt": "Bạn là giáo viên Tiếng Anh THCS chuyên tạo câu hỏi luyện tập. Hãy luôn tạo đầy đủ số lượng câu hỏi theo yêu cầu.",
            "question_type": "tiếng Anh"
        },
        "physics": {
            "name": "Vật lý",
            "system_prompt": "Bạn là giáo viên Vật lý THCS chuyên tạo câu hỏi luyện tập. Hãy luôn tạo đầy đủ số lượng câu hỏi theo yêu cầu.",
            "question_type": "vật lý"
        },
        "chemistry": {
            "name": "Hóa học",
            "system_prompt": "Bạn là giáo viên Hóa học THCS chuyên tạo câu hỏi luyện tập. Hãy luôn tạo đầy đủ số lượng câu hỏi theo yêu cầu.",
            "question_type": "hóa học"
        },
        "biology": {
            "name": "Sinh học",
            "system_prompt": "Bạn là giáo viên Sinh học THCS chuyên tạo câu hỏi luyện tập. Hãy luôn tạo đầy đủ số lượng câu hỏi theo yêu cầu.",
            "question_type": "sinh học"
        },
        "geography": {
            "name": "Địa lý",
            "system_prompt": "Bạn là giáo viên Địa lý THCS chuyên tạo câu hỏi luyện tập. Hãy luôn tạo đầy đủ số lượng câu hỏi theo yêu cầu.",
            "question_type": "địa lý"
        },
        "history": {
            "name": "Lịch sử",
            "system_prompt": "Bạn là giáo viên Lịch sử THCS chuyên tạo câu hỏi luyện tập. Hãy luôn tạo đầy đủ số lượng câu hỏi theo yêu cầu.",
            "question_type": "lịch sử"
        },
        "civics": {
            "name": "Giáo dục Công dân",
            "system_prompt": "Bạn là giáo viên Giáo dục Công dân THCS chuyên tạo câu hỏi luyện tập. Hãy luôn tạo đầy đủ số lượng câu hỏi theo yêu cầu.",
            "question_type": "giáo dục công dân"
        },
        "informatics": {
            "name": "Tin học",
            "system_prompt": "Bạn là giáo viên Tin học THCS chuyên tạo câu hỏi luyện tập. Hãy luôn tạo đầy đủ số lượng câu hỏi theo yêu cầu.",
            "question_type": "tin học"
        }
    }
    
    # Kiểm tra subject hợp lệ
    if request.subject not in subject_config:
        return {
            "success": False,
            "error": f"Môn học '{request.subject}' không hợp lệ. Các môn học hỗ trợ: {', '.join(subject_config.keys())}"
        }
    
    try:
        config = subject_config[request.subject]
        recent_tests_text = "\n".join([f"- {test}" for test in request.recent_tests])
        
        # Xử lý questionTypes nếu có
        question_types_text = ""
        if hasattr(request, 'questionTypes') and request.questionTypes:
            question_types_text = f"\n\nLoại câu hỏi cần tạo:\n" + "\n".join([f"- {qtype}" for qtype in request.questionTypes])
        
        prompt = f"""Dựa trên các chủ đề {config['question_type']} sau đây, hãy tạo ra một câu hỏi {config['question_type']} cho mỗi chủ đề.

Môn học: {config['name']}

Các chủ đề:
{recent_tests_text}{question_types_text}

QUY TẮC QUAN TRỌNG:
- Trả về ĐÚNG {len(request.recent_tests)} câu hỏi tương ứng với {len(request.recent_tests)} chủ đề
- Câu hỏi phải phù hợp với môn {config['name']} và chương trình THCS
- CHỈ trả về JSON array, KHÔNG có text giải thích thêm
- KHÔNG có dấu phẩy thừa sau phần tử cuối
- Format chính xác như sau:

[
    {{
        "topic": "<tên chủ đề y nguyên>",
        "question": "<câu hỏi {config['question_type']} liên quan>",
        "difficulty": "<easy|medium|hard>"
    }},
    {{
        "topic": "<tên chủ đề y nguyên>",
        "question": "<câu hỏi {config['question_type']} liên quan>",
        "difficulty": "<easy|medium|hard>"
    }}
]
"""
        
        response = client.chat.completions.create(
            model=MODEL_NAME,
            messages=[
                {"role": "system", "content": config['system_prompt']},
                {"role": "user", "content": prompt}
            ],
            max_tokens=2048,
            temperature=0.7,
            top_p=0.9
        )
        
        response_text = response.choices[0].message.content
        
        # Parse JSON từ response
        quiz_result = extract_json_from_text(response_text)
        
        if quiz_result and isinstance(quiz_result, list):
            return {
                "success": True,
                "questions": quiz_result,
                "topics": request.recent_tests,
                "subject": request.subject,
                "subject_name": config['name']
            }
        else:
            return {
                "success": True,
                "questions": [],
                "raw_response": response_text,
                "topics": request.recent_tests,
                "subject": request.subject,
                "subject_name": config['name']
            }
    except Exception as e:
        return {
            "success": False,
            "error": str(e),
            "subject": request.subject
        }

@app.post("/analyze-teacher-feedback")
async def analyze_teacher_feedback(request: TeacherFeedbackRequest):
    """
    Phân tích đánh giá của giáo viên và trả về câu hỏi bài tập + gợi ý cải thiện cho tất cả các môn học
    """
    # Map Vietnamese subject names to English keys
    subject_name_map = {
        "Toán": "math",
        "Ngữ văn": "van",
        "Tiếng Anh": "english",
        "Vật lý": "physics",
        "Hóa học": "chemistry",
        "Sinh học": "biology",
        "Địa lý": "geography",
        "Lịch sử": "history",
        "Giáo dục Công dân": "civics",
        "Tin học": "informatics"
    }
    
    # Convert Vietnamese subject name to English key if needed
    subject_key = subject_name_map.get(request.subject, request.subject.lower())
    
    # Định nghĩa config cho từng môn học
    subject_config = {
        "math": {
            "name": "Toán",
            "system_prompt": "Bạn là giáo viên Toán THCS chuyên phân tích năng lực học sinh. CHỈ trả về JSON, không có text khác.",
            "example": {
                "exercise_question": "Giải phương trình: 2x + 5 = 15",
                "improve_suggestion": "Em cần rèn luyện thêm kỹ năng chuyển vế và tính toán cẩn thận hơn."
            }
        },
        "van": {
            "name": "Ngữ văn",
            "system_prompt": "Bạn là giáo viên Ngữ văn THCS chuyên phân tích năng lực học sinh. CHỈ trả về JSON, không có text khác.",
            "example": {
                "exercise_question": "Nêu cảm nhận của em về nhân vật trong đoạn trích đã học.",
                "improve_suggestion": "Em cần phân biệt rõ nội dung và nghệ thuật trong bài phân tích."
            }
        },
        "english": {
            "name": "Tiếng Anh",
            "system_prompt": "Bạn là giáo viên Tiếng Anh THCS chuyên phân tích năng lực học sinh. CHỈ trả về JSON, không có text khác.",
            "example": {
                "exercise_question": "Rewrite the sentence using past perfect tense: She finished her homework before dinner.",
                "improve_suggestion": "Em cần ôn lại cấu trúc thì quá khứ hoàn thành và cách sử dụng trong ngữ cảnh."
            }
        },
        "physics": {
            "name": "Vật lý",
            "system_prompt": "Bạn là giáo viên Vật lý THCS chuyên phân tích năng lực học sinh. CHỈ trả về JSON, không có text khác.",
            "example": {
                "exercise_question": "Tính lực ma sát khi một vật có khối lượng 5kg trượt trên mặt phẳng ngang với hệ số ma sát 0.3.",
                "improve_suggestion": "Em cần nắm vững công thức tính lực ma sát và đơn vị đo lường."
            }
        },
        "chemistry": {
            "name": "Hóa học",
            "system_prompt": "Bạn là giáo viên Hóa học THCS chuyên phân tích năng lực học sinh. CHỈ trả về JSON, không có text khác.",
            "example": {
                "exercise_question": "Cân bằng phương trình phản ứng: Fe + O₂ → Fe₂O₃",
                "improve_suggestion": "Em cần rèn luyện kỹ năng cân bằng phương trình hóa học và hiểu rõ quy tắc hóa trị."
            }
        },
        "biology": {
            "name": "Sinh học",
            "system_prompt": "Bạn là giáo viên Sinh học THCS chuyên phân tích năng lực học sinh. CHỈ trả về JSON, không có text khác.",
            "example": {
                "exercise_question": "Giải thích quá trình quang hợp ở thực vật và vai trò của diệp lục.",
                "improve_suggestion": "Em cần hiểu rõ các giai đoạn quang hợp và mối liên hệ giữa chúng."
            }
        },
        "geography": {
            "name": "Địa lý",
            "system_prompt": "Bạn là giáo viên Địa lý THCS chuyên phân tích năng lực học sinh. CHỈ trả về JSON, không có text khác.",
            "example": {
                "exercise_question": "Phân tích đặc điểm khí hậu nhiệt đới gió mùa ở miền Nam Việt Nam.",
                "improve_suggestion": "Em cần nắm vững các yếu tố ảnh hưởng đến khí hậu và cách phân tích bản đồ khí hậu."
            }
        },
        "history": {
            "name": "Lịch sử",
            "system_prompt": "Bạn là giáo viên Lịch sử THCS chuyên phân tích năng lực học sinh. CHỈ trả về JSON, không có text khác.",
            "example": {
                "exercise_question": "Phân tích ý nghĩa của cuộc khởi nghĩa Hai Bà Trưng trong lịch sử dân tộc.",
                "improve_suggestion": "Em cần nắm rõ mốc thời gian và nguyên nhân - kết quả của các sự kiện lịch sử."
            }
        },
        "civics": {
            "name": "Giáo dục Công dân",
            "system_prompt": "Bạn là giáo viên Giáo dục Công dân THCS chuyên phân tích năng lực học sinh. CHỈ trả về JSON, không có text khác.",
            "example": {
                "exercise_question": "Trình bày các quyền và nghĩa vụ cơ bản của công dân theo Hiến pháp 2013.",
                "improve_suggestion": "Em cần hiểu rõ sự khác biệt giữa quyền và nghĩa vụ công dân trong các tình huống cụ thể."
            }
        },
        "informatics": {
            "name": "Tin học",
            "system_prompt": "Bạn là giáo viên Tin học THCS chuyên phân tích năng lực học sinh. CHỈ trả về JSON, không có text khác.",
            "example": {
                "exercise_question": "Viết chương trình nhập vào số nguyên n và in ra tổng các số từ 1 đến n.",
                "improve_suggestion": "Em cần rèn luyện tư duy thuật toán và cách sử dụng vòng lặp hiệu quả."
            }
        }
    }
    
    # Kiểm tra subject hợp lệ
    if subject_key not in subject_config:
        return {
            "success": False,
            "error": f"Môn học '{request.subject}' không hợp lệ. Các môn học hỗ trợ: {', '.join(subject_config.keys())} hoặc tên tiếng Việt"
        }
    
    try:
        config = subject_config[subject_key]
        subject_name = config["name"]
        system_prompt = config["system_prompt"]
        example = config["example"]
        
        # Format teacher comments - handle both string and list
        if isinstance(request.teacher_comment, list):
            comments_text = "\n".join([f"- {comment}" for comment in request.teacher_comment])
        else:
            # If it's a string, keep it as is
            comments_text = request.teacher_comment
        
        prompt = f"""Dựa trên nhận xét của giáo viên về bài học "{request.lesson}" môn {subject_name}, hãy tạo câu hỏi bài tập và gợi ý cải thiện cho học sinh.

Nhận xét của giáo viên:
{comments_text}

Bài học: {request.lesson}
Môn học: {subject_name}

YÊU CẦU:
- Tạo câu hỏi bài tập phù hợp với nội dung bài học và nhận xét của giáo viên
- Đưa ra gợi ý cải thiện cụ thể dựa trên điểm yếu trong nhận xét
- Câu hỏi phải có độ khó vừa phải, phù hợp với trình độ THCS
- Gợi ý phải thiết thực và có thể áp dụng được

Trả về JSON với format SAU (KHÔNG thêm text khác):
{{"exercise_question": "<câu hỏi bài tập {subject_name}>", "improve_suggestion": "<gợi ý cải thiện cụ thể>"}}

Ví dụ cho môn {subject_name}:
{{"exercise_question": "{example['exercise_question']}", "improve_suggestion": "{example['improve_suggestion']}"}}"""
        
        response = client.chat.completions.create(
            model=MODEL_NAME,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": prompt}
            ],
            max_tokens=512,
            temperature=0.3,
            top_p=0.8,
            response_format={"type": "json_object"}
        )
        
        response_text = response.choices[0].message.content
        
        # Parse JSON từ response
        feedback_result = extract_json_from_text(response_text)
        
        # Validate JSON structure
        if feedback_result and isinstance(feedback_result, dict):
            if "exercise_question" in feedback_result and "improve_suggestion" in feedback_result:
                if isinstance(feedback_result.get("exercise_question"), str) and isinstance(feedback_result.get("improve_suggestion"), str):
                    return {
                        "success": True,
                        "result": feedback_result,
                        "teacher_comment": request.teacher_comment,
                        "subject": subject_name,
                        "lesson": request.lesson
                    }
        
        return {
            "success": False,
            "error": "Model không tạo được JSON hợp lệ. Vui lòng thử lại.",
            "raw_response": response_text,
            "teacher_comment": request.teacher_comment
        }
    
    except Exception as e:
        return {
            "success": False,
            "error": str(e)
        }


@app.post("/recent-test-grading")
async def recent_test_grading(request: RecentTestGradingRequest):
    """
    Chấm điểm một nhóm câu hỏi dựa trên rubric toàn cục cho môn học
    Subjects: math, van, english, physics, chemistry, biology, geography, history, civics, informatics
    """
    # Định nghĩa config cho từng môn học
    subject_config = {
        "math": {"name": "Toán", "system_prompt": "Bạn là giáo viên Toán THCS chuyên chấm điểm bài tập. CHỈ trả về JSON, không có text khác."},
        "van": {"name": "Ngữ văn", "system_prompt": "Bạn là giáo viên Ngữ văn THCS chuyên chấm điểm bài tập. CHỈ trả về JSON, không có text khác."},
        "english": {"name": "Tiếng Anh", "system_prompt": "Bạn là giáo viên Tiếng Anh THCS chuyên chấm điểm bài tập. CHỈ trả về JSON, không có text khác."},
        "physics": {"name": "Vật lý", "system_prompt": "Bạn là giáo viên Vật lý THCS chuyên chấm điểm bài tập. CHỈ trả về JSON, không có text khác."},
        "chemistry": {"name": "Hóa học", "system_prompt": "Bạn là giáo viên Hóa học THCS chuyên chấm điểm bài tập. CHỈ trả về JSON, không có text khác."},
        "biology": {"name": "Sinh học", "system_prompt": "Bạn là giáo viên Sinh học THCS chuyên chấm điểm bài tập. CHỈ trả về JSON, không có text khác."},
        "geography": {"name": "Địa lý", "system_prompt": "Bạn là giáo viên Địa lý THCS chuyên chấm điểm bài tập. CHỈ trả về JSON, không có text khác."},
        "history": {"name": "Lịch sử", "system_prompt": "Bạn là giáo viên Lịch sử THCS chuyên chấm điểm bài tập. CHỈ trả về JSON, không có text khác."},
        "civics": {"name": "Giáo dục Công dân", "system_prompt": "Bạn là giáo viên Giáo dục Công dân THCS chuyên chấm điểm bài tập. CHỈ trả về JSON, không có text khác."},
        "informatics": {"name": "Tin học", "system_prompt": "Bạn là giáo viên Tin học THCS chuyên chấm điểm bài tập. CHỈ trả về JSON, không có text khác."}
    }
    
    # Kiểm tra subject hợp lệ
    if request.subject not in subject_config:
        return {
            "success": False,
            "error": f"Môn học '{request.subject}' không hợp lệ. Các môn học hỗ trợ: {', '.join(subject_config.keys())}"
        }
    
    try:
        config = subject_config[request.subject]
        
        # Lấy rubric cho môn học
        subject_vietnamese = SUBJECT_MAPPING.get(request.subject, "")
        rubric_criteria = GLOBAL_RUBRICS.get(subject_vietnamese, [])
        
        # Format rubric text
        rubric_text = ""
        if rubric_criteria:
            rubric_text = "\n\nTiêu chí đánh giá (Rubric):\n"
            for criterion in rubric_criteria:
                rubric_text += f"- {criterion['name']}: {criterion['weight']}%\n"
        
        # Tạo danh sách câu hỏi để chấm
        questions_text = ""
        for i, q in enumerate(request.questions, 1):
            questions_text += f"\n{i}. Câu hỏi: {q['question']}\n"
            questions_text += f"   Chủ đề: {q['topic']}\n"
            questions_text += f"   Độ khó: {q['difficulty']}\n"
            questions_text += f"   Câu trả lời của học sinh: {q['student_answer']}\n"
        
        prompt = f"""Bạn là giáo viên {config['name']} THCS. Hãy chấm điểm {len(request.questions)} câu hỏi sau theo rubric đã cho.

Môn học: {config['name']}{rubric_text}

Danh sách câu hỏi và câu trả lời của học sinh:{questions_text}

YÊU CẦU CHẤM:
- Đánh giá MỖI câu hỏi dựa trên độ chính xác, logic và phương pháp giải
- Áp dụng tiêu chí rubric để đánh giá toàn diện
- Với mỗi câu: xác định đúng/sai (isCorrect), cho điểm (0-10), và nhận xét chi tiết
- Điểm phải phản ánh chính xác mức độ đạt được theo từng tiêu chí rubric
- Nếu câu trả lời đúng về bản chất toán học → isCorrect = true
- Chỉ đánh giá isCorrect = false nếu kết quả hoặc logic sai rõ ràng

TRẢ VỀ DUY NHẤT JSON array (KHÔNG có text khác, KHÔNG dùng markdown):
[
  {{
    "question_number": 1,
    "isCorrect": <true || false>,
    "score": <điểm từ 0-10>,
    "comments": "Nhận xét chi tiết về bài làm, bao gồm: 1) Đánh giá độ chính xác, 2) Phân tích các tiêu chí rubric, 3) Điểm mạnh/yếu",
    "correct_answer": "Đáp án đúng và lời giải chi tiết"
  }},
  ...
]

Lưu ý: Phải trả về ĐÚNG {len(request.questions)} kết quả chấm điểm."""
        
        response = client.chat.completions.create(
            model=MODEL_NAME,
            messages=[
                {"role": "system", "content": config['system_prompt']},
                {"role": "user", "content": prompt}
            ],
            max_tokens=2048,
            temperature=0.3,
            top_p=0.9
        )
        
        response_text = response.choices[0].message.content
        grading_results = extract_json_from_text(response_text)
        
        # Validate response
        if grading_results and isinstance(grading_results, list) and len(grading_results) == len(request.questions):
            # Combine results with original questions
            detailed_results = []
            for i, (question_data, grading_data) in enumerate(zip(request.questions, grading_results)):
                detailed_results.append({
                    "question_number": i + 1,
                    "question": question_data["question"],
                    "student_answer": question_data["student_answer"],
                    "topic": question_data["topic"],
                    "difficulty": question_data["difficulty"],
                    "isCorrect": grading_data.get("isCorrect", False),
                    "score": grading_data.get("score", 0),
                    "comments": grading_data.get("comments", ""),
                    "correct_answer": grading_data.get("correct_answer", "")
                })
            
            # Calculate overall statistics
            total_score = sum([r["score"] for r in detailed_results])
            average_score = total_score / len(detailed_results)
            correct_count = sum([1 for r in detailed_results if r["isCorrect"]])
            
            return {
                "success": True,
                "subject": request.subject,
                "subject_name": config['name'],
                "total_questions": len(request.questions),
                "correct_count": correct_count,
                "average_score": round(average_score, 2),
                "rubric_criteria": rubric_criteria,
                "detailed_results": detailed_results
            }
        else:
            return {
                "success": False,
                "error": "Model không trả về đủ kết quả chấm điểm hoặc format không đúng.",
                "raw_response": response_text,
                "expected_count": len(request.questions),
                "received_count": len(grading_results) if isinstance(grading_results, list) else 0
            }
    
    except Exception as e:
        return {
            "success": False,
            "error": str(e)
        }

@app.post("/performance/question-generation")
async def performance_question_generation(request: PerformanceQuestionRequest):
    """
    Tạo câu hỏi luyện tập hàng ngày dựa trên hiệu suất học tập gần đây của học sinh
    Trả về JSON theo format dailyPracticeQuestion của student schema
    """
    # Định nghĩa thông tin cho từng môn học
    subject_config = {
        "math": {
            "name": "Toán",
            "system_prompt": "Bạn là giáo viên Toán THCS chuyên phân tích năng lực học sinh. CHỈ trả về JSON, không có text khác."
        },
        "van": {
            "name": "Ngữ văn",
            "system_prompt": "Bạn là giáo viên Ngữ văn THCS chuyên phân tích năng lực học sinh. CHỈ trả về JSON, không có text khác."
        },
        "english": {
            "name": "Tiếng Anh",
            "system_prompt": "Bạn là giáo viên Tiếng Anh THCS chuyên phân tích năng lực học sinh. CHỈ trả về JSON, không có text khác."
        },
        "physics": {
            "name": "Vật lý",
            "system_prompt": "Bạn là giáo viên Vật lý THCS chuyên phân tích năng lực học sinh. CHỈ trả về JSON, không có text khác."
        },
        "chemistry": {
            "name": "Hóa học",
            "system_prompt": "Bạn là giáo viên Hóa học THCS chuyên phân tích năng lực học sinh. CHỈ trả về JSON, không có text khác."
        },
        "biology": {
            "name": "Sinh học",
            "system_prompt": "Bạn là giáo viên Sinh học THCS chuyên phân tích năng lực học sinh. CHỈ trả về JSON, không có text khác."
        },
        "geography": {
            "name": "Địa lý",
            "system_prompt": "Bạn là giáo viên Địa lý THCS chuyên phân tích năng lực học sinh. CHỈ trả về JSON, không có text khác."
        },
        "history": {
            "name": "Lịch sử",
            "system_prompt": "Bạn là giáo viên Lịch sử THCS chuyên phân tích năng lực học sinh. CHỈ trả về JSON, không có text khác."
        },
        "civics": {
            "name": "Giáo dục Công dân",
            "system_prompt": "Bạn là giáo viên Giáo dục Công dân THCS chuyên phân tích năng lực học sinh. CHỈ trả về JSON, không có text khác."
        },
        "informatics": {
            "name": "Tin học",
            "system_prompt": "Bạn là giáo viên Tin học THCS chuyên phân tích năng lực học sinh. CHỈ trả về JSON, không có text khác."
        }
    }
    
    # Kiểm tra subject hợp lệ
    if request.subject not in subject_config:
        return {
            "success": False,
            "error": f"Môn học '{request.subject}' không hợp lệ. Các môn học hỗ trợ: {', '.join(subject_config.keys())}"
        }
    
    try:
        config = subject_config[request.subject]
        
        # Tạo thông tin về các bài test gần đây
        test_info_text = ""
        if request.recent_tests and len(request.recent_tests) > 0:
            test_info_text = "\n\nThông tin các bài kiểm tra gần đây:\n"
            for i, test in enumerate(request.recent_tests[:3], 1):  # Chỉ lấy 3 bài gần nhất
                test_info_text += f"{i}. Bài: {test.get('title', 'N/A')} - Điểm: {test.get('score', 0)}/10\n"
        else:
            test_info_text = "\n\nHọc sinh chưa có kết quả kiểm tra gần đây."
        
        # Tạo prompt dựa trên hiệu suất
        avg_score = 0
        if request.recent_tests and len(request.recent_tests) > 0:
            scores = [test.get('score', 0) for test in request.recent_tests if test.get('score')]
            avg_score = sum(scores) / len(scores) if scores else 0
        
        # Xác định độ khó phù hợp
        if avg_score >= 8:
            difficulty_guidance = "Tạo câu hỏi ở mức độ NÂNG CAO để thách thức và phát triển năng lực học sinh xuất sắc này."
        elif avg_score >= 6:
            difficulty_guidance = "Tạo câu hỏi ở mức độ TRUNG BÌNH để củng cố kiến thức và nâng cao dần năng lực."
        else:
            difficulty_guidance = "Tạo câu hỏi ở mức độ CƠ BẢN để giúp học sinh nắm vững kiến thức nền tảng."
        
        prompt = f"""Dựa trên thông tin hiệu suất học tập của học sinh môn {config['name']}, hãy tạo MỘT câu hỏi luyện tập phù hợp.

{test_info_text}

Điểm trung bình: {avg_score:.1f}/10

Hướng dẫn: {difficulty_guidance}

YÊU CẦU:
1. Phân tích điểm yếu/mạnh của học sinh dựa trên điểm số
2. Đề xuất gợi ý cải thiện cụ thể
3. Tạo câu hỏi phù hợp với trình độ hiện tại

Trả về 3 JSON với format SAU (KHÔNG thêm text khác, KHÔNG dùng markdown):
{{
  "question": "Nội dung câu hỏi luyện tập chi tiết và rõ ràng",
  "answer": "Câu trả lời mẫu đầy đủ, có hướng dẫn từng bước",
  "ai_score": 0,
  "improvement_suggestions": "Gợi ý cải thiện dựa trên điểm yếu được phát hiện từ các bài test gần đây, bao gồm: 1) Điểm cần cải thiện, 2) Phương pháp học tập đề xuất, 3) Kỹ năng cần rèn luyện"
}}

Lưu ý: 
- ai_score luôn là 0 (sẽ được cập nhật sau khi học sinh làm bài)
- improvement_suggestions phải CỤ THỂ và DỰA TRÊN hiệu suất thực tế
- Câu hỏi phải PHÙ HỢP với chương trình THCS"""
        
        response = client.chat.completions.create(
            model=MODEL_NAME,
            messages=[
                {"role": "system", "content": config['system_prompt']},
                {"role": "user", "content": prompt}
            ],
            max_tokens=1024,
            temperature=0.7,
            top_p=0.9,
            response_format={"type": "json_object"}
        )
        
        response_text = response.choices[0].message.content
        result = extract_json_from_text(response_text)
        
        # Validate JSON structure
        if result and isinstance(result, dict):
            # Kiểm tra các trường bắt buộc
            required_fields = ["question", "answer", "improvement_suggestions"]
            if all(field in result for field in required_fields):
                # Đảm bảo ai_score = 0
                result["ai_score"] = 0
                
                # Validate kiểu dữ liệu
                if (isinstance(result.get("question"), str) and 
                    isinstance(result.get("answer"), str) and
                    isinstance(result.get("improvement_suggestions"), str)):
                    
                    return {
                        "success": True,
                        "question": result["question"],
                        "answer": result["answer"],
                        "ai_score": 0,
                        "improvement_suggestions": result["improvement_suggestions"],
                        "subject": request.subject,
                        "average_score": round(avg_score, 2)
                    }
        
        return {
            "success": False,
            "error": "Model không tạo được JSON hợp lệ với đầy đủ các trường bắt buộc. Vui lòng thử lại.",
            "subject": request.subject
        }
    
    except Exception as e:
        return {
            "success": False,
            "error": str(e),
            "subject": request.subject
        }


@app.post("/grade-with-rubric")
async def grade_with_rubric(request: RubricGradingRequest):
    """
    Chấm điểm bài tập dựa trên rubric do giáo viên cung cấp.
    Trả về điểm chi tiết theo từng tiêu chí và tổng điểm.
    """
    try:
        # Lấy tên môn học tiếng Việt
        subject_vn = SUBJECT_MAPPING.get(request.subject, request.subject)
        
        # Chuẩn bị thông tin rubric
        rubric_info = ""
        for i, criteria in enumerate(request.rubric_criteria, 1):
            name = criteria.get('name', f'Tiêu chí {i}')
            weight = criteria.get('weight', 0)
            description = criteria.get('description', '')
            rubric_info += f"{i}. {name} (Trọng số: {weight}%)"
            if description:
                rubric_info += f" - {description}"
            rubric_info += "\\n"
        
        # Chuẩn bị thông tin câu hỏi và câu trả lời
        qa_info = ""
        for i, qa in enumerate(request.questions_and_answers, 1):
            qa_info += f"""
Câu {i}:
- Đề bài: {qa.get('question', 'N/A')}
- Loại câu hỏi: {qa.get('questionType', 'N/A')}
- Điểm tối đa: {qa.get('grade', 0)}
- Đáp án mẫu: {qa.get('solution', 'N/A')}
- Bài làm của học sinh: {qa.get('studentAnswer', 'Chưa trả lời')}
"""
        
        # Tạo prompt cho AI
        grading_prompt = f"""Bạn là giáo viên {subject_vn} THCS. Hãy chấm điểm bài làm của học sinh "{request.student_name}" dựa trên rubric sau.

📋 THÔNG TIN BÀI KIỂM TRA:
- Tên bài: {request.test_title}
- Môn học: {subject_vn}

📊 RUBRIC ĐÁNH GIÁ:
{rubric_info}

📝 NỘI DUNG BÀI LÀM:
{qa_info}

🎯 YÊU CẦU:
1. Chấm điểm từng tiêu chí trong rubric (0-10 điểm cho mỗi tiêu chí)
2. Tính điểm theo trọng số: Điểm tiêu chí × (Trọng số/100)
3. Tổng điểm = Tổng các điểm đã tính trọng số
4. Nhận xét chi tiết cho từng tiêu chí
5. Nhận xét tổng thể và gợi ý cải thiện

Trả về JSON với format sau (KHÔNG thêm text khác):
{{
    "rubric_scores": [
        {{
            "criteria_name": "Tên tiêu chí",
            "weight": <trọng số>,
            "score": <điểm 0-10>,
            "weighted_score": <điểm đã nhân trọng số>,
            "comment": "Nhận xét cho tiêu chí này"
        }}
    ],
    "question_scores": [
        {{
            "question_number": <số thứ tự câu>,
            "max_score": <điểm tối đa>,
            "student_score": <điểm học sinh đạt được>,
            "is_correct": <true/false>,
            "feedback": "Nhận xét cho câu này"
        }}
    ],
    "total_score": <tổng điểm cuối cùng (0-10)>,
    "overall_comment": "Nhận xét tổng thể về bài làm",
    "strengths": ["Điểm mạnh 1", "Điểm mạnh 2"],
    "weaknesses": ["Điểm yếu 1", "Điểm yếu 2"],
    "improvement_suggestions": "Gợi ý cải thiện chi tiết"
}}"""

        response = client.chat.completions.create(
            model=MODEL_NAME,
            messages=[
                {"role": "system", "content": f"Bạn là giáo viên {subject_vn} THCS chuyên nghiệp. Chấm điểm công bằng, chi tiết và có tính xây dựng. CHỈ trả về JSON, không có text khác."},
                {"role": "user", "content": grading_prompt}
            ],
            max_tokens=2048,
            temperature=0.3,
            top_p=0.9,
            response_format={"type": "json_object"}
        )
        
        response_text = response.choices[0].message.content
        grading_result = extract_json_from_text(response_text)
        
        if grading_result and isinstance(grading_result, dict):
            # Validate và đảm bảo các trường cần thiết
            if "total_score" not in grading_result:
                # Tính tổng điểm từ rubric_scores nếu không có
                if "rubric_scores" in grading_result:
                    total = sum(item.get("weighted_score", 0) for item in grading_result["rubric_scores"])
                    grading_result["total_score"] = round(total, 2)
                else:
                    grading_result["total_score"] = 0
            
            return {
                "success": True,
                "grading_result": grading_result,
                "test_title": request.test_title,
                "subject": request.subject,
                "student_name": request.student_name
            }
        
        return {
            "success": False,
            "error": "Không thể parse kết quả chấm điểm từ AI. Vui lòng thử lại.",
            "raw_response": response_text
        }
        
    except Exception as e:
        return {
            "success": False,
            "error": str(e),
            "test_title": request.test_title,
            "subject": request.subject
        }