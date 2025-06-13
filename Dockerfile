FROM python:3.9-slim

# تثبيت libGL
RUN apt-get update && apt-get install -y libgl1-mesa-glx

WORKDIR /app

COPY requirements.txt .

RUN pip install --upgrade pip
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

CMD ["gunicorn", "-w", "1", "-b", "0.0.0.0:5000", "SimulationFinalJune:app"]
