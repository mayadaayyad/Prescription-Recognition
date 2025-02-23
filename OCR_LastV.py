import spacy
import easyocr
import re
import cv2
import numpy as np
from PIL import Image, ImageEnhance, ImageFilter
from rapidfuzz import process
import pandas as pd
from transformers import pipeline


nlp = spacy.load("en_core_sci_lg")
ner_pipeline = pipeline("ner", model="d4data/biomedical-ner-all")

dataset_path = r"C:\Users\NOUR SOFT\Desktop\MedicineDataSet.csv"
df = pd.read_csv(dataset_path)

drug_list = df["Medicine Name"].tolist()  # قائمة أسماء الأدوية

# تحسين الصورـ مسح النص العربي فقط

def preprocess_image(image_path):
    image = cv2.imread(image_path)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    # مسح النص العربي باستخدام التعرف على الحروف
    reader = easyocr.Reader(['ar', 'en'])
    text_results = reader.readtext(gray, detail=1)
    
    for (bbox, text, prob) in text_results:
        if re.search(r'[؀-ۿ]', text):  # التأكد من أن النص عربي فعلًا
            x_min, y_min = map(int, bbox[0])
            x_max, y_max = map(int, bbox[2])
            cv2.rectangle(gray, (x_min, y_min), (x_max, y_max), (255, 255, 255), -1)  # مسح النص العربي فقط
    
    # تحسين الصورة بعد المسح
    image_pil = Image.fromarray(gray)
    image_pil = image_pil.filter(ImageFilter.SHARPEN)
    enhancer = ImageEnhance.Contrast(image_pil)
    image_pil = enhancer.enhance(3.0)
    return image_pil

# دالة لاستخراج النص من صورة الروشتة باستخدام EasyOCR
def extract_text_from_image(image_path):
    processed_image = preprocess_image(image_path)
    reader = easyocr.Reader(['en'])
    result = reader.readtext(np.array(processed_image), detail=0)
    text = " ".join(result)  # إضافة سطر جديد بين كل عنصر
    return text

# دالة تنظيف النصوص
def clean_text(text):
    text = text.lower()
    text = re.sub(r"[^a-zA-Z0-9 ./%-]", "", text)  # إزالة الرموز الغريبة مع الاحتفاظ بالنقاط والشرطات والرموز المهمة
    return text

# دالة لاستخراج الأدوية من النص باستخدام NER 
def extract_drugs(text):
    ner_results = ner_pipeline(text)
    drugs_ner = [entity['word'] for entity in ner_results if 'entity' in entity and "medication" in entity['entity'].lower()]
    return list(set(drugs_ner))  # إزالة التكرارات

# دالة لتصحيح أسماء الأدوية باستخدام الداتاسيت
def correct_drug_names(extracted_drugs, drug_list):
    corrected_drugs = []
    for drug in extracted_drugs:
        match = process.extractOne(drug, drug_list, score_cutoff=80)
        if match:
            corrected_drugs.append(match[0])
    return corrected_drugs

# تجربة الكود على صورة روشتة
def process_prescription(image_path):
    extracted_text = extract_text_from_image(image_path)
    cleaned_text = clean_text(extracted_text)
    extracted_drugs = extract_drugs(cleaned_text)
    corrected_drugs = correct_drug_names(extracted_drugs, drug_list)

    print("Extracted Text :\n", extracted_text)


# استخدام الكود على صورة معينة
image_path = r"C:\Users\NOUR SOFT\Desktop\drug1.jpg"  
process_prescription(image_path)