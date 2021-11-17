# churn_prediction_with_pyspark

## İş Problemi
Şirketi terk edecek müşterileri tahmin 
edebilecek bir makine öğrenmesi modeli 
geliştirilmesi beklenmektedir.

##  Veri Seti Hikayesi
10000 gözlemden ve 12 değişkenden oluşmaktadır.
Bağımsız değişkenler müşterilere ilişkin bilgiler barındırmaktadır.
Bağımlı değişken müşteri terk durumunu ifade etmektedir.

## Değişkenler
- Surname – Soy isim
- CreditScore – Kredi skoru
- Geography – Ülke (Germany/France/Spain)
- Gender – Cinsiyet
- Age – Yaş
- Tenure – Kaç yıllık müşteri olduğu bilgisi
- NumOfProducts – Kullanılan banka ürünü
- HasCrCard – Kredi kartı durumu (0=No,1=Yes)
- IsActiveMember – Aktif üyelik durumu (0=No,1=Yes)
- EstimatedSalary – Tahmini maaş
- Exited: – Terk mi değil mi? (0=No,1=Yes)
