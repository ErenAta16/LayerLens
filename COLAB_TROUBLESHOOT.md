# Colab Troubleshooting Guide

## İlk Hücre Sorunları ve Çözümleri

### Sorun 1: Cython Derleme Hatası

**Hata:**
```
ERROR: Failed building wheel for layerlens
```

**Çözüm - Alternatif Hücre 1:**
```python
# Önce build tools yükle
!apt-get update
!apt-get install -y build-essential

# Repository'yi klonla
!git clone https://github.com/ErenAta16/LayerLens.git
%cd LayerLens

# Cython'u önce yükle
!pip install Cython>=3.0 -q

# Sonra paketi yükle
!pip install -e ".[demo,yolo,pipeline]" -q

print("✅ LayerLens kuruldu!")
```

### Sorun 2: NumPy/SciPy Versiyon Uyumsuzluğu

**Hata:**
```
ERROR: numpy>=2.0.0 is required but you have numpy 1.x.x
```

**Çözüm:**
```python
# NumPy ve SciPy'yi önce yükle
!pip install "numpy>=1.24,<2.0" "scipy>=1.10.0" -q

# Sonra LayerLens
!git clone https://github.com/ErenAta16/LayerLens.git
%cd LayerLens
!pip install -e ".[demo,yolo,pipeline]" -q
```

### Sorun 3: Git Clone Hatası

**Hata:**
```
fatal: destination path 'LayerLens' already exists
```

**Çözüm:**
```python
# Eski klasörü sil ve yeniden klonla
import shutil
import os

if os.path.exists('LayerLens'):
    shutil.rmtree('LayerLens')

!git clone https://github.com/ErenAta16/LayerLens.git
%cd LayerLens
!pip install -e ".[demo,yolo,pipeline]" -q
```

### Sorun 4: Memory Hatası (Büyük Modeller)

**Hata:**
```
RuntimeError: CUDA out of memory
```

**Çözüm:**
```python
# GPU memory'yi temizle
import torch
import gc

torch.cuda.empty_cache()
gc.collect()

# Batch size'ı küçült
batch_size = 1  # Varsayılan 4 yerine
```

### Sorun 5: Import Hatası

**Hata:**
```
ModuleNotFoundError: No module named 'layerlens'
```

**Çözüm:**
```python
# Python path'i kontrol et
import sys
print("Python path:", sys.path)

# Paketi yeniden yükle
!pip install -e . --force-reinstall -q

# Import'u test et
try:
    import layerlens
    print("✅ Import başarılı!")
except Exception as e:
    print(f"❌ Import hatası: {e}")
    !pip list | grep layerlens
```

## Güvenli Kurulum Hücresi (Tüm Sorunları Önler)

```python
# Güvenli kurulum - tüm olası sorunları önler
import os
import shutil

# 1. Eski klasörü temizle
if os.path.exists('LayerLens'):
    shutil.rmtree('LayerLens')

# 2. Build tools yükle (gerekirse)
try:
    !apt-get update -qq
    !apt-get install -y build-essential -qq
except:
    pass  # Zaten yüklü olabilir

# 3. Temel bağımlılıkları yükle
!pip install "numpy>=1.24,<2.0" "scipy>=1.10.0" "Cython>=3.0" -q

# 4. Repository'yi klonla
!git clone https://github.com/ErenAta16/LayerLens.git
%cd LayerLens

# 5. Paketi yükle
!pip install -e ".[demo,yolo,pipeline]" -q

# 6. Kurulumu doğrula
try:
    from layerlens.pipeline import run_pipeline
    print("✅ LayerLens başarıyla kuruldu ve test edildi!")
except Exception as e:
    print(f"❌ Kurulum hatası: {e}")
    print("\nManuel kontrol:")
    !pip list | grep -E "(layerlens|numpy|scipy|cython)"
```

## Hızlı Doğrulama Hücresi

İlk hücreden sonra bu hücreyi çalıştırın:

```python
# Kurulum doğrulama
import sys

print("=" * 70)
print("KURULUM DOĞRULAMA")
print("=" * 70)

# 1. Python versiyonu
print(f"Python: {sys.version.split()[0]}")

# 2. LayerLens import testi
try:
    import layerlens
    print("✅ layerlens import: BAŞARILI")
except Exception as e:
    print(f"❌ layerlens import: BAŞARISIZ - {e}")

# 3. Alt modül testleri
modules_to_test = [
    "layerlens.pipeline",
    "layerlens.config",
    "layerlens.models",
    "layerlens.profiling",
    "layerlens.optimization",
]

for module in modules_to_test:
    try:
        __import__(module)
        print(f"✅ {module}: BAŞARILI")
    except Exception as e:
        print(f"❌ {module}: BAŞARISIZ - {e}")

# 4. Cython modülleri testi
try:
    from layerlens.profiling._aggregators import aggregate_scores_c
    print("✅ Cython modülleri: DERLENMİŞ")
except:
    print("⚠️  Cython modülleri: DERLENMEMİŞ (Python fallback kullanılacak)")

# 5. Bağımlılık kontrolü
try:
    import numpy
    import scipy
    import torch
    print(f"✅ NumPy: {numpy.__version__}")
    print(f"✅ SciPy: {scipy.__version__}")
    print(f"✅ PyTorch: {torch.__version__}")
except Exception as e:
    print(f"❌ Bağımlılık hatası: {e}")

print("=" * 70)
```

## Yaygın Uyarılar (Sorun Değil)

Bu uyarılar normaldir ve çalışmayı engellemez:

1. **Cython FutureWarning**: Cython derleme uyarıları
2. **Dependency conflicts**: Versiyon uyumsuzlukları (genelde çalışır)
3. **Git warnings**: Repository zaten var uyarıları
4. **Build warnings**: Derleme sırasında uyarılar

## Kritik Hatalar (Düzeltme Gerekli)

Bu hatalar düzeltilmelidir:

1. **ModuleNotFoundError**: Paket kurulmamış
2. **ImportError**: Modül bulunamıyor
3. **CompilationError**: Cython derlenemiyor
4. **CUDA errors**: GPU sorunları

Bu hatalardan birini görürseniz, yukarıdaki çözümleri deneyin.

