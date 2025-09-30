# Camera Calibration and 3D Projection with Stereo Vision

Bu proje, **OpenCV** kullanarak tek ve stereo kamera kalibrasyonu, 3D nesne projeksiyonu, homografi hesaplama ve derinlik haritası oluşturmayı sağlar. Ayrıca, 3D küp projeksiyonu ve disparity tabanlı derinlik haritası görselleştirmesi içerir.

---

## Özellikler

1. **Tek Kamera Kalibrasyonu**  
   - Satranç tahtası kullanarak kamera içsel ve distorsiyon parametrelerini hesaplar.

2. **Stereo Kamera Kalibrasyonu**  
   - İki kamera için eşleşen görüntüler üzerinden stereo parametreleri (R, T, E, F) hesaplar.  
   - Stereo rectification ve Q matrisi üretir.

3. **3D Nesne Projeksiyonu**  
   - 3D koordinatlardaki nesneleri 2D görüntüye projekte eder.  
   - Örnek: 3D küp projeksiyonu.

4. **Homografi Hesaplama ve Uygulama**  
   - İki nokta seti arasındaki dönüşümü hesaplar ve görüntüye uygular.

5. **Disparity ve Derinlik Haritası**  
   - Stereo görüntülerden disparity hesaplar.  
   - Q matrisi kullanarak derinlik haritası oluşturur.

---

## Gereksinimler

- Python 3.x
- OpenCV (`opencv-python`)
- NumPy
- Matplotlib
- Glob (Python standart kütüphanesi)

Kurulum için:

```bash
pip install opencv-python numpy matplotlib

project_root/
│
├─ calibration_images/     # Tek kamera kalibrasyon görüntüleri
├─ left_images/            # Sol kamera stereo görüntüleri
├─ right_images/           # Sağ kamera stereo görüntüleri
├─ camera_system.py        # Ana Python kodu
└─ README.md               # Proje açıklaması
