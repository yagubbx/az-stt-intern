import os
import torch
import pandas as pd
import jiwer
import re
import librosa
import warnings
warnings.filterwarnings("ignore")
from transformers import pipeline

def clean_text(text):
    text = str(text).lower()
    text = re.sub(r'[^\w\səöğşçıü]', '', text)
    return text.strip()

def main():
    print("ASR Baza tətbiqi işə düşdü...")
    
    # manual qeyd etdiyim ses fayllari: (wav formatinda)
    my_dataset = {
        "1.wav": "mənim ixtisasım süni intellekt mühəndisliyidir",
        "2.wav": "mənim komputerim cox güclü işləyir",
        "3.wav": "mən bu tapşiriği uğurla tamamlamaq istəyirəm",
        "4.wav": "speech recognition real dünyada geniş istifadə olunur",
        "5.wav": "mən data mühəndisliyi ilə məşğulam",
        "6.wav": "mən öz uzərimdə çox çalışıram",
        "7.wav": "süni intellekt insan nitqini analiz edə bilir",
        "8.wav": "bu model səsi mətnə çevirir",
        "9.wav": "mən tələbəyəm",
        "10.wav": "maşın öyrənməsi çox güclü sahədir",
        "11.wav": "proqramlaşdırma mənim üçün maraqlıdır",
        "12.wav": "mən süni intellekt akademiyasında təhsil alıram",
        "13.wav": "mən avtomatik nitq tanıma sistemi hazırlayıram",
        "14.wav": "bu model müxtəlif səsləri tanımağa çalışır"
    }
    
    audio_folder = "my_audio"
    
    if not os.path.exists(audio_folder):
        print(f"'{audio_folder}' qovluğu yoxdur. yaradin ve sesleri icine elave edin. (wav formatinda)")
        return

    # Modeli yukleyirik (komp-a yuklemisem deye tez yuklenecek)
    MODEL_NAME = "nijatzeynalov/wav2vec2-large-mms-1b-azerbaijani-common_voice15.0" 
    print(f"Model kesden(cache) yuklenir ({MODEL_NAME})...")
    
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    asr_pipeline = pipeline("automatic-speech-recognition", model=MODEL_NAME, device=device)
    
    results = []
    
    for file_name, reference_sentence in my_dataset.items():
        audio_path = os.path.join(audio_folder, file_name)
        
        if not os.path.exists(audio_path):
            print(f"Diqqet! Tapilmadi: {audio_path}")
            continue
            
        print(f"Emal olunur: {file_name}...")
        
        audio_array, _ = librosa.load(audio_path, sr=16000)
        reference_text = clean_text(reference_sentence)
        
        prediction = asr_pipeline(audio_array)["text"]
        prediction_text = clean_text(prediction)
        
        try:
            wer_score = jiwer.wer(reference_text, prediction_text)
            cer_score = jiwer.cer(reference_text, prediction_text)
        except ValueError:
            wer_score, cer_score = 1.0, 1.0
            
        results.append({
            "File": file_name,
            "Reference": reference_text,
            "Prediction": prediction_text,
            "WER": wer_score,
            "CER": cer_score
        })

    df = pd.DataFrame(results)
    if not df.empty:
        avg_wer = df["WER"].mean() * 100
        avg_cer = df["CER"].mean() * 100
        
        print("-" * 40)
        print(f"📊 Neticeler:")
        print(f"Ortalama WER: {avg_wer:.2f}%")
        print(f"Ortalama CER: {avg_cer:.2f}%")
        print("-" * 40)
        
        df_sorted = df.sort_values(by="WER")
        print("\nƏn Yaxşı 5 Nümunə:")
        for _, row in df_sorted.head(5).iterrows():
            print(f"  Ref : {row['Reference']}\n  Pred: {row['Prediction']}\n  WER : {row['WER']:.2f}\n")
            
        print("Ən Pis 5 Nümunə:")
        for _, row in df_sorted.tail(5).iterrows():
            print(f"  Ref : {row['Reference']}\n  Pred: {row['Prediction']}\n  WER : {row['WER']:.2f}\n")
        
        os.makedirs("results", exist_ok=True)
        df.to_csv("results/metrics.csv", index=False)
        print("SON olaraq: Bütün nəticələr 'results/metrics.csv' faylına saxlanıldı.")

if __name__ == "__main__":
    main()