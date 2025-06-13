# استخدم صورة رسمية من Python 3.9
FROM python:3.9-slim

# تعيين مجلد العمل داخل الحاوية
WORKDIR /app

# نسخ ملف المتطلبات وتثبيت الحزم
COPY requirements.txt .
RUN pip install --upgrade pip
RUN pip install --no-cache-dir -r requirements.txt

# نسخ بقية ملفات المشروع
COPY . .

# تحديد البورت الذي سيعمل عليه Flask
EXPOSE 5000

# أمر التشغيل (يفترض أن ملفك فيه app = Flask(__name__))
CMD ["gunicorn", "--bind", "0.0.0.0:5000", "SimulationFinalJune:app"]
