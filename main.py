import numpy as np
import matplotlib.pyplot as plt
import cv2
# --- 1. CHUẨN BỊ ẢNH ĐẦU VÀO ---
f = cv2.imread('Assets/Flowers.jpg', 0)
if f is None:
    raise FileNotFoundError('Không tìm được ảnh')
f = f / 255
# --- 2. CHUYỂN SANG MIỀN TẦN SỐ ---
F = np.fft.fft2(f)
F = np.fft.fftshift(F)
# --- 3. XÂY DỰNG HÀM TRUYỀN ĐẠT ---
P, Q = F.shape
H = np.zeros((P, Q), dtype=np.float32)
for u in range(P):
    for v in range(Q):
        H[u, v] = -4*np.pi*np.pi*((u-P/2)**2 + (v-Q/2)**2)
# --- 4. TẠO ẢNH LAPLACE ---
lap = H * F
lap = np.fft.ifftshift(lap)
lap = np.fft.ifft2(lap)
lap = np.real(lap)
# --- 5. TĂNG CƯỜNG VÀ CHUẨN HOÁ LẠI ---
old_range = np.max(lap) - np.min(lap)
new_range = 2
lap_scaled = (((lap - np.min(lap)) * new_range) / old_range) -1

c = -1
g = f + c*lap_scaled
g = np.clip(g, 0, 1)
# --- 6. HIỂN THỊ KẾT QUẢ ---
plt.figure(figsize=(12, 12), dpi=150)
# 1. Ảnh Gốc
plt.subplot(2, 2, 1)
plt.imshow(f, cmap='gray')
plt.title('H1. Ảnh Gốc', fontsize=20)
plt.axis('off')
# 2. Phổ Tần Số
plt.subplot(2, 2, 2)
plt.imshow(np.log1p(np.abs(F)), cmap='gray')
plt.title('H2. Phổ Tần Số', fontsize=20)
plt.axis('off')
# 3. Ảnh Laplace
plt.subplot(2, 2, 3)
plt.imshow(lap_scaled, cmap='gray')
plt.title('H3. Ảnh Laplace', fontsize=20)
plt.axis('off')
# 4. Ảnh Tăng Cường
plt.subplot(2, 2, 4)
plt.imshow(g, cmap='gray')
plt.title('H4. Ảnh Tăng Cường' , fontsize=20)
plt.axis('off')

plt.tight_layout()
plt.show()