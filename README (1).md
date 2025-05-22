
# GoldQuant Dashboard

لوحة تحكم احترافية مبنية بـ Streamlit لتحليل سوق الذهب باستخدام:
- التحليل الفني (SMA, RSI)
- النماذج الإحصائية (ARIMA)
- محاكاة مونت كارلو للمخاطر
- إشارات تداول ذكية باستخدام XGBoost

## كيفية التشغيل محليًا

```bash
pip install -r requirements.txt
streamlit run gold_quant_dashboard.py
```

## النشر على الإنترنت (Streamlit Cloud)

1. ارفع الملفات إلى مستودع GitHub عام.
2. ادخل على [https://streamlit.io/cloud](https://streamlit.io/cloud)
3. اختر المستودع، وحدد `gold_quant_dashboard.py` كملف رئيسي.
4. اضغط Deploy.

ستحصل على رابط عام لتطبيقك.

---

## الملفات

- `gold_quant_dashboard.py`: التطبيق الرئيسي
- `requirements.txt`: مكتبات بايثون المطلوبة
- `README.md`: دليل الاستخدام والنشر
