# Otonom Araç Navigasyon Sistemi için SVM Tabanlı Güvenlik Modülü 🚗🛡️

Bu proje, otonom araçların çevresindeki engelleri (Tehlikeli Engel vs. Güvenli Bölge) birbirinden en güvenli şekilde ayırabilmesi için **Hard-Margin SVM (Support Vector Machines)** algoritmasını kullanarak bir karar sınırı (hiper düzlem) oluşturur.

## 🎯 Projenin Amacı
Otonom bir aracın sensörlerinden gelen koordinat verilerini kullanarak, iki sınıf arasında **maksimum marjin (güvenlik koridoru)** bırakacak şekilde matematiksel bir model eğitir. Bu sayede araç, sadece engelleri tanımakla kalmaz, onlara en güvenli mesafede duracağı rotayı belirler.

## 🛠️ Teknik Özellikler
- **Programlama Dili:** Java
- **Algoritma:** Sequential Minimal Optimization (SMO) - John Platt (1998)
- **Model:** Hard-Margin Support Vector Machine
- **Mimari:** Layered Architecture (Sunum, Uygulama, Algoritma, Veri katmanları)
- **Optimizasyon:** Quadratic Programming (QP) Dual Problem çözümü

## 📐 Matematiksel Temel
Proje, aşağıdaki optimizasyon problemini temel alır:
- **Primal Problem:** $\frac{1}{2} \|w\|^2$ değerini minimize ederek marjini ($2/\|w\|$) maksimize etmek.
- **Kısıtlar:** $y_i(w \cdot x_i + b) \geq 1$ (Tüm noktaların doğru tarafta ve marjinin dışında olması).
- **Dual Problem:** Lagrange çarpanları ($\alpha$) kullanılarak verilerin iç çarpımları (Kernel Matrix) üzerinden çözüm sağlanır.

## 🏗️ Yazılım Mimarisi (SOLID & OOP)
- **Encapsulation:** Veri noktaları ve model parametreleri `Immutable` (değişmez) yapılarla korunur.
- **SRP (Single Responsibility):** Eğitim mantığı (`SVMTrainer`), çözüm mantığı (`QuadraticProgrammingSolver`) ve veri modeli (`SVMModel`) birbirinden ayrılmıştır.
- **Memory Management:** Java GC uyumlu, manuel bellek yönetimi gerektirmeyen verimli kernel matrisi kullanımı.

## 🚀 Örnek Çıktı (Demo)
Sistem eğitildiğinde aşağıdaki gibi bir sonuç üretir:
```text
[SMO] 2 iterasyonda yakınsandı.
[TRAIN] 2 destek vektörü bulundu.
[TRAIN] ||w|| = 0,342997, Marjin = 5,830952

Karar Sınırı Denklemi:
-0,2941·x₁ + -0,1765·x₂ + (1,6471) = 0
